import argparse
import json
import time

import jax
import numpy as np
import optax
import jax.numpy as jnp
from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        # network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}", devices.shape[1])
        # print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))
        
        batch_size = 16
        token_size = 1024
        key = jax.random.PRNGKey(42)
        for i in range(2):
            tokens = jax.random.uniform(key,minval=0,maxval=1e5,shape=(batch_size,token_size)).astype(jnp.uint32)

            start = time.time()

            padded_tokens = jnp.pad(tokens, ((0,0),(0,token_size))).astype(jnp.uint32)
            length = jnp.ones(batch_size, dtype=jnp.uint32) * token_size

            output = network.generate(padded_tokens, length, token_size*2, {"top_p": jnp.ones(batch_size) * 0.9,
                                                                    "temp": jnp.ones(batch_size) * 0.75})
            print(output[1][0][:, :, 0].shape)
            print(f"completion done in {time.time() - start:06}s") 
            #Takes 264s in second iteration for batch_size 16 and token_size 1024 (a generation length of 2048)
            