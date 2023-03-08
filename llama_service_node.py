# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from llama_service.srv import LlamaService, LlamaServiceResponse
import rospy
# run: torchrun --nproc_per_node 1 llama_service_node.py
class LlamaServiceClass:
    def __init__(self):
        self.load_model()
        rospy.init_node('llama_service_node')
        s = rospy.Service('llama_service', LlamaService, self.run_req)
        print("Ready to run llama.")
        rospy.spin()

    @staticmethod
    def setup_model_parallel() -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)
        return local_rank, world_size

    @staticmethod
    def load(
        ckpt_dir: str,
        tokenizer_path: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        max_batch_size: int,
    ) -> LLaMA:
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

        generator = LLaMA(model, tokenizer)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return generator

    def load_model(
        self,
        ckpt_dir: str = "./llama_model/7B",
        tokenizer_path: str = "./llama_model/tokenizer.model",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_seq_len: int = 512,
        max_batch_size: int = 32,
    ):
        self.temperature = temperature
        self.top_p = top_p
        local_rank, world_size = self.setup_model_parallel()
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")
        self.generator = self.load(
            ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
        )

    def run_req(self, req):
        print(f"input is {req.prompt}")
        prompts = [req.prompt]
        results = self.generator.generate(
            prompts, max_gen_len=256, temperature=self.temperature, top_p=self.top_p
        )
        for result in results:
            print(result)
            print("\n==================================\n")
        return LlamaServiceResponse(result)


if __name__ == "__main__":
    LlamaServiceClass()
