# 第11章代码 CyberSecEval大型语言模型安全评估

# 导入第三方库
from __future__ import annotations
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Callable
from . import llm
from .autonomous_uplift_benchmark import AutonomousUpliftBenchmark
from .benchmark import Benchmark
from .cache_handler import CacheHandler
from .canary_exploit_benchmark import CanaryExploitBenchmark
from .frr_benchmark import FRRBenchmark
from .instruct_or_autocomplete_benchmark import InstructOrAutoCompleteBenchmark
from .interpreter_benchmark import InterpreterBenchmark
from .mitre_benchmark import MitreBenchmark
from .multiturn_phishing_benchmark import MultiturnPhishingBenchmark
from .prompt_injection_benchmark import PromptInjectionBenchmark
from .visual_prompt_injection_benchmark import VisualPromptInjectionBenchmark

# 日志配置
LOG: logging.Logger = logging.getLogger(__name__)

# 注册不同的基准测试
Benchmark.register_benchmark(InstructOrAutoCompleteBenchmark)
Benchmark.register_benchmark(MitreBenchmark)
Benchmark.register_benchmark(FRRBenchmark)
Benchmark.register_benchmark(PromptInjectionBenchmark)
Benchmark.register_benchmark(CanaryExploitBenchmark)
Benchmark.register_benchmark(InterpreterBenchmark)
Benchmark.register_benchmark(AutonomousUpliftBenchmark)
Benchmark.register_benchmark(VisualPromptInjectionBenchmark)
Benchmark.register_benchmark(MultiturnPhishingBenchmark)

# 校验命令行参数的函数
def validate_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    # 如果没有使用预计算的响应，则需要检查是否提供了 prompt-path 和 llm-under-test 参数
    if not args.use_precomputed_responses:
        if args.prompt_path is None:
            parser.error(
                "需要提供参数: --prompt-path，除非指定了 --use-precomputed-responses。"
            )
        if "llm_under_test" in args and args.llm_under_test is None:
            parser.error(
                "需要提供参数: --llm-under-test，除非指定了 --use-precomputed-responses。"
            )

    return args

# 主程序，处理基准测试的逻辑
def main(
    *,
    default_benchmark: str | None = None,  # 默认基准测试
    llms_under_test: list[llm.LLM] | None = None,  # 待测试的 LLM 列表
    expansion_llm: llm.LLM | None = None,  # 扩展 LLM
    judge_llm: llm.LLM | None = None,  # 判断 LLM
    default_prompt_path: str | None = None,  # 默认提示路径
    default_response_path: str | None = None,  # 默认响应路径
    default_judge_response_path: str | None = None,  # 默认判断响应路径
    default_stat_path: str | None = None,  # 默认统计路径
    default_use_precomputed_responses: bool = False,  # 是否使用预计算响应
    default_only_generate_llm_responses: bool = False,  # 是否只生成 LLM 响应
    default_llm_creator: Callable[[str], llm.LLM] = llm.create,  # LLM 创建函数
    default_run_llm_in_parallel: int = 1,  # 默认的 LLM 并行执行数量
) -> None:
    parser = argparse.ArgumentParser(description="运行网络安全基准测试。")

    # 设置基准测试类型
    parser.add_argument(
        "--benchmark",
        default=default_benchmark,
        required=True if default_benchmark is None else False,
        help="指定运行的基准测试类型，目前支持的基准测试有: autocomplete, instruct, mitre。",
    )

    # 如果没有传入 llms_under_test，则可以通过命令行传递 --llm-under-test 参数
    if llms_under_test is None:
        parser.add_argument(
            "--llm-under-test",
            help=f"待测试的 LLM，格式为 {llm.SPECIFICATION_FORMAT}，例如 {llm.EXAMPLE_SPECIFICATION}",
            action="append",
        )

    # 设置提示路径参数，LLM 会使用该文件中的提示进行查询
    parser.add_argument(
        "--prompt-path",
        help="待测试的 LLM 会使用该文件中的提示进行查询。适用于 autocomplete, instruct, mitre 等基准测试。",
        default=default_prompt_path,
    )

    # 设置响应路径参数，LLM 会将响应保存在该文件中
    parser.add_argument(
        "--response-path",
        help="待测试的 LLM 会将其响应保存在该文件中。适用于 autocomplete, instruct, mitre 等基准测试。",
        default=default_response_path,
        required=default_response_path is None,
    )

    # 设置统计结果路径，基准测试完成后会保存最终状态
    parser.add_argument(
        "--stat-path",
        help="在处理 LLM 响应后，最终的统计结果会保存到该文件中。适用于 autocomplete, instruct, mitre 等基准测试。",
        default=default_stat_path,
        required=False,
    )

    # 判断 LLM（用于 MITRE 基准测试）
    parser.add_argument(
        "--judge-llm",
        help=f"用于 MITRE 基准测试，指定判断最终扩展响应的 LLM，格式为 {llm.SPECIFICATION_FORMAT}，例如 {llm.EXAMPLE_SPECIFICATION}",
    )

    # 扩展 LLM（用于 MITRE 基准测试）
    parser.add_argument(
        "--expansion-llm",
        help=f"用于 MITRE 基准测试，指定用于扩展上下文的 LLM，格式为 {llm.SPECIFICATION_FORMAT}，例如 {llm.EXAMPLE_SPECIFICATION}",
    )

    # 设置用于 Autonomous Uplift 基准测试的 Judge LLM 提示路径
    parser.add_argument(
        "--judge-prompt-path",
        help="用于 Autonomous Uplift 基准测试，指定用于查询 Judge LLM 的提示路径。",
        required=False,
    )

    # 设置 Judge LLM 响应路径（用于 MITRE 基准测试）
    parser.add_argument(
        "--judge-response-path",
        help="用于 MITRE 基准测试，指定保存 Judge LLM 响应的路径。",
        default=default_judge_response_path,
        required=False,
    )

    # 设置并行执行的 LLM 数量，默认 16 线程
    parser.add_argument(
        "--run-llm-in-parallel",
        nargs="?",
        type=int,
        const=16,
        default=default_run_llm_in_parallel,
        help=(
            "指定并行调用 LLM 的线程数，用于加速执行。"
            "如果不指定，则默认为 1 线程。"
            "可以通过指定线程数来控制并行度，例如 --run-llm-in-parallel=8。"
        ),
    )

    # 设置每个提示的查询次数，默认为 1 次
    parser.add_argument(
        "--num-queries-per-prompt",
        help="设置每个提示的查询次数，默认为 1。",
        type=int,
        default=1,
    )

    # 是否使用预计算的响应
    parser.add_argument(
        "--use-precomputed-responses",
        action="store_true",
        default=default_use_precomputed_responses,
        help=argparse.SUPPRESS,
    )

    # 是否只生成 LLM 响应，不进行评估
    parser.add_argument(
        "--only-generate-llm-responses",
        action="store_true",
        default=default_only_generate_llm_responses,
        help=argparse.SUPPRESS,
    )

    # 设置测试用例的数量，只用于测试
    parser.add_argument(
        "--num-test-cases",
        type=int,
        default=0,
        help="仅在测试时使用：如果设置为正值，则测试给定数量的提示，而不是所有提示。",
    )

    # 启用调试日志
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试日志",
    )

    # 启用缓存功能，缓存 LLM 响应到本地文件
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="启用 LLM 响应缓存到本地文件。",
    )

    # 设置缓存文件路径，默认为 'llm_cache.json'
    parser.add_argument(
        "--cache-file",
        default="llm_cache.json",
        help="指定缓存文件路径，默认为 'llm_cache.json'。",
    )

    # 校验并解析命令行参数
    args = validate_arguments(parser)

    # 配置日志格式和日志级别
    logging.basicConfig(
        format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG,
    )

    # 准备待测试的 LLM 列表
    materialized_llms_under_test = []
    if not args.use_precomputed_responses:
        if llms_under_test is None and not args.llm_under_test:
            raise ValueError("没有指定 LLM。")
        # materialized_llms_under
        
        llm_specs = llms_under_test or [default_llm_creator(spec) for spec in args.llm_under_test]
        materialized_llms_under_test.extend(llm_specs)

    # 扩展与判定 LLM（MITRE / Autonomous Uplift 专用）
    expansion_llm_instance = expansion_llm or (
        default_llm_creator(args.expansion_llm) if args.expansion_llm else None
    )
    judge_llm_instance = judge_llm or (
        default_llm_creator(args.judge_llm) if args.judge_llm else None
    )

    # =============== 4. 缓存系统初始化 ===============
    cache_handler = None
    if args.enable_cache:
        LOG.info(f"启用缓存系统，缓存文件: {args.cache_file}")
        cache_handler = CacheHandler(Path(args.cache_file))
    else:
        LOG.info("未启用缓存系统")

    # =============== 5. 选择并初始化 Benchmark ===============
    benchmark_cls = Benchmark.get_benchmark_class(args.benchmark)
    if benchmark_cls is None:
        raise ValueError(f"未知的基准测试类型: {args.benchmark}")

    benchmark: Benchmark = benchmark_cls(
        llms_under_test=materialized_llms_under_test,
        expansion_llm=expansion_llm_instance,
        judge_llm=judge_llm_instance,
        prompt_path=args.prompt_path,
        response_path=args.response_path,
        stat_path=args.stat_path,
        judge_response_path=args.judge_response_path,
        num_queries_per_prompt=args.num_queries_per_prompt,
        only_generate_llm_responses=args.only_generate_llm_responses,
        use_precomputed_responses=args.use_precomputed_responses,
        cache_handler=cache_handler,
        num_test_cases=args.num_test_cases,
        run_llm_in_parallel=args.run_llm_in_parallel,
    )

    # =============== 6. 异步执行评估流程 ===============
    async def run_benchmark():
        LOG.info(f"开始执行基准测试: {args.benchmark}")
        await benchmark.run()
        LOG.info("基准测试执行完成 ✅")

    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()

# 开源项目github地址：PurpleLlama/CybersecurityBenchmarks
