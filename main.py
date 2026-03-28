"""LLM Roofline Simulation - 入口文件"""

import argparse
import json
from pathlib import Path
from llm_sim.configs import HardwareConfig, ModelConfig, QuantConfig, DeploymentConfig
from llm_sim.inference.models import DecodeDeepSeekV32, PrefillDeepSeekV32


def run_simulation(hardware_path, model_path, quant_path, deploy_path, output_dir="outputs"):
    print("=" * 70)
    print("LLM Roofline Simulation - 大模型性能仿真工具")
    print("=" * 70)

    # 加载配置
    print("\n[1/3] 加载配置文件...")
    hardware = HardwareConfig.from_json(hardware_path)
    model = ModelConfig.from_json(model_path)
    quant = QuantConfig.from_json(quant_path)
    deploy = DeploymentConfig.from_json(deploy_path)

    print(f"  硬件: {hardware.chip_name} x{hardware.total_chips} ({hardware.total_chips // hardware.chips_per_card}卡 x {hardware.chips_per_card}芯片/卡)")
    print(f"  模型: {model.model_type} - {model.num_hidden_layers}层, {model.hidden_size}维")
    if model.is_moe:
        print(f"        MoE: {model.num_experts}专家, top-{model.num_experts_per_tok}, shared={getattr(model, 'num_shared_experts', 0)}")
    if getattr(model, 'attention_type', '') == 'mla':
        print(f"        Attention: MLA (kv_lora_rank={getattr(model, 'kv_lora_rank', 512)})")
    print(f"  量化: weight={quant.default_weight_bits}bit, act={quant.default_activation_compute_bits}bit")
    print(f"  部署: mode={deploy.deployment_mode}, TP={deploy.attention_tp}, EP={deploy.expert_parallel}, PP={deploy.pipeline_parallel}")
    print(f"        MTP={deploy.mtp_length}, accept={deploy.mtp_acceptance_rate}")
    print(f"        CP={deploy.context_parallel}")
    prefix_hit = getattr(deploy, 'prefix_cache_hit_rate', 0.0)
    if prefix_hit > 0:
        print(f"        PrefixCache={prefix_hit*100:.0f}%")
    print(f"  负载: input={deploy.input_length}, output={deploy.output_length}")

    # 计算TP组数量和全局batch size
    num_tp_groups = max(1, hardware.total_chips // deploy.attention_tp // deploy.context_parallel)
    total_bs = deploy.micro_batch_size * num_tp_groups
    num_cards = hardware.total_chips // hardware.chips_per_card
    print(f"  Batch: micro={deploy.micro_batch_size}, total_bs={total_bs} (TP组={num_tp_groups}, 卡数={num_cards})")

    # 根据 deployment_mode 选择模型
    print("\n[2/3] 运行性能仿真...")
    if deploy.deployment_mode == "decode":
        inference_model = DecodeDeepSeekV32(hardware, model, deploy, quant)
    elif deploy.deployment_mode == "prefill":
        inference_model = PrefillDeepSeekV32(hardware, model, deploy, quant)
    elif deploy.deployment_mode == "pd":
        # TODO: 实现 PD 混合模型
        raise NotImplementedError(f"PD 混合模型尚未实现")
    else:
        raise ValueError(f"不支持的 deployment_mode: {deploy.deployment_mode}，支持的模式: prefill, decode, pd")

    # 保存结果
    print("\n[3/3] 保存结果...")
    profiling = inference_model.save_results(output_dir)

    # 读取单卡和系统级性能
    single_card_path = Path(output_dir) / "single_card_perf.json"
    system_perf_path = Path(output_dir) / "system_perf.json"

    single_card = None
    system_perf = None
    if single_card_path.exists():
        with open(single_card_path, 'r', encoding='utf-8') as f:
            single_card = json.load(f)
    if system_perf_path.exists():
        with open(system_perf_path, 'r', encoding='utf-8') as f:
            system_perf = json.load(f)

    # 显示关键指标
    print(f"\n{'='*70}")
    print(f"【Per-Request时延】")
    print(f"  总时延:     {profiling['total_time_ms']:.3f} ms")
    if profiling['ttft_ms'] is not None:
        print(f"  TTFT:       {profiling['ttft_ms']:.3f} ms/request")
    if profiling['tpot_ms'] is not None:
        print(f"  TPOT:       {profiling['tpot_ms']:.3f} ms/token")
    print(f"  内存占用:   {profiling['memory_usage_gb']:.2f} GB")
    print(f"\n  单层Block:  {profiling['single_block_time_ms']:.4f} ms")
    print(f"  单层Attn:   {profiling['single_attn_time_ms']:.4f} ms")
    print(f"  单层FFN:    {profiling['single_ffn_time_ms']:.4f} ms")

    bd = profiling['latency_breakdown']
    rt = profiling['latency_ratio']
    print(f"\n  时延分解:")
    print(f"    CUBE:   {bd['cube_time_ms']:.4f} ms  ({rt['cube_ratio']*100:.1f}%)")
    print(f"    Vector: {bd['vector_time_ms']:.4f} ms  ({rt['vector_ratio']*100:.1f}%)")
    print(f"    Memory: {bd['mem_time_ms']:.4f} ms  ({rt['mem_ratio']*100:.1f}%)")
    print(f"    Comm:   {bd['comm_time_ms']:.4f} ms  ({rt['comm_ratio']*100:.1f}%)")

    # 显示单卡性能
    if single_card:
        print(f"\n【单卡性能】({single_card['num_tp_groups'] // single_card['num_cards']} TP组/卡)")
        print(f"  QPS:        {single_card['qps_per_card']:.2f} req/s")
        print(f"  TPS:        {single_card['tps_per_card']:.2f} tokens/s")
        if single_card.get('ttft_ms') is not None and single_card.get('tpot_ms') is None:
            print(f"  公式: Prefill时 QPS = total_bs / TTFT = {single_card['total_bs']} / {single_card['ttft_ms']:.3f}ms = {single_card['qps_per_card']:.2f}")
        else:
            print(f"  公式: Decode时 QPS = TPS / output_length = {single_card['tps_per_card']:.2f} / {single_card['output_length']} = {single_card['qps_per_card']:.2f}")

    # 显示系统级性能
    if system_perf:
        print(f"\n【系统级性能】({system_perf['num_cards']}卡 x {system_perf['chips_per_card']}芯片, total_bs={system_perf['total_bs']})")
        print(f"  System QPS: {system_perf['system_qps']:.2f} req/s")
        print(f"  System TPS: {system_perf['system_tps']:.2f} tokens/s")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LLM Roofline Simulation")
    parser.add_argument("--hardware", default="configs/hardware/ascend_910c_16card.json")
    parser.add_argument("--model", default="configs/models/deepseek_v3_2.json")
    parser.add_argument("--quant", default="configs/quantization/default_fp8.json")
    parser.add_argument("--deploy", default="configs/deployment/pd_separated_8k.json")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    run_simulation(args.hardware, args.model, args.quant, args.deploy, args.output)


if __name__ == "__main__":
    main()
