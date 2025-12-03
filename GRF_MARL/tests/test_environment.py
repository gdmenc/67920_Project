"""
Environment and dependency tests.
Tests core dependencies, imports, and GFootball environment.
"""

import sys
import os

from .conftest import TestResults, print_status, print_subsection, PROJECT_ROOT


def test_python_version(results: TestResults):
    """Check Python version."""
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 8
    results.add(print_status("Python version", ok, f"{v.major}.{v.minor}.{v.micro}"), "Python version")
    return ok


def test_core_imports(results: TestResults, verbose=False):
    """Test core Python dependencies."""
    deps = [
        ("numpy", "numpy"),
        ("ray", "ray"),
        ("torch", "torch"),
        ("gym", "gym"),
        ("omegaconf", "omegaconf"),
        ("yaml", "pyyaml"),
        ("tensorboard", "tensorboard"),
        ("nashpy", "nashpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("pydantic", "pydantic"),
        ("wandb", "wandb"),
        ("colorlog", "colorlog"),
        ("tabulate", "tabulate"),
        ("easydict", "easydict"),
        ("tree", "dm-tree"),
        ("grpc", "grpcio"),
        ("psutil", "psutil"),
    ]
    
    all_ok = True
    for module, pkg_name in deps:
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "?")
            ok = print_status(pkg_name, True, ver if verbose else "")
            results.add(ok)
        except ImportError as e:
            ok = print_status(pkg_name, False, str(e))
            results.add(ok, pkg_name)
            all_ok = False
    
    return all_ok


def test_pytorch_cuda(results: TestResults, verbose=False):
    """Test PyTorch CUDA support."""
    import torch
    
    all_ok = True
    all_ok &= print_status("PyTorch version", True, torch.__version__)
    results.add(True)
    
    cuda_available = torch.cuda.is_available()
    all_ok &= print_status("CUDA available", cuda_available)
    results.add(cuda_available, "CUDA available")
    
    if cuda_available:
        print_status("CUDA version", True, torch.version.cuda)
        results.add(True)
        
        cudnn_ver = torch.backends.cudnn.version()
        print_status("cuDNN version", True, str(cudnn_ver) if cudnn_ver else "Not available")
        results.add(True)
        
        gpu_count = torch.cuda.device_count()
        print_status("GPU count", True, str(gpu_count))
        results.add(True)
        
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print_status(f"GPU {i}", True, f"{name} (compute {cap[0]}.{cap[1]})")
            results.add(True)
        
        # Test basic CUDA operations
        try:
            x = torch.randn(100, 100, device='cuda')
            y = x @ x.T
            z = torch.nn.functional.softmax(y, dim=-1)
            del x, y, z
            torch.cuda.empty_cache()
            print_status("CUDA tensor ops", True)
            results.add(True)
        except Exception as e:
            print_status("CUDA tensor ops", False, str(e))
            results.add(False, "CUDA tensor ops")
            all_ok = False
    
    return all_ok


def test_gfootball(results: TestResults, verbose=False):
    """Test Google Research Football environment."""
    try:
        import gfootball.env as football_env
        print_status("gfootball import", True)
        results.add(True)
        
        # Try creating a simple environment
        env = football_env.create_environment(
            env_name="academy_empty_goal_close",
            representation="simple115v2",
            number_of_left_players_agent_controls=1,
            stacked=False,
            logdir=None,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False
        )
        obs = env.reset()
        print_status("gfootball env creation", True, f"obs shape: {obs.shape}")
        results.add(True)
        
        # Test a step
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])
        print_status("gfootball env step", True)
        results.add(True)
        
        env.close()
        return True
    except Exception as e:
        print_status("gfootball", False, str(e))
        results.add(False, "gfootball")
        return False


def test_light_malib_imports(results: TestResults, verbose=False):
    """Test all major light_malib imports."""
    imports = [
        ("light_malib.utils.logger", "Logger"),
        ("light_malib.utils.cfg", "Config utils"),
        ("light_malib.utils.random", "Random utils"),
        ("light_malib.utils.timer", "Timer utils"),
        ("light_malib.utils.distributed", "Distributed utils"),
        ("light_malib.envs.gr_football.env", "GRFootball env wrapper"),
        ("light_malib.envs.gr_football.encoders.encoder_basic", "Basic encoder"),
        ("light_malib.algorithm.mappo.policy", "MAPPO policy"),
        ("light_malib.algorithm.mappo.trainer", "MAPPO trainer"),
        ("light_malib.algorithm.mappo.loss", "MAPPO loss"),
        ("light_malib.algorithm.qmix.policy", "QMIX policy"),
        ("light_malib.algorithm.dqn.policy", "DQN policy"),
        ("light_malib.buffer.data_server", "Data server"),
        ("light_malib.buffer.policy_server", "Policy server"),
        ("light_malib.training.training_manager", "Training manager"),
        ("light_malib.training.distributed_trainer", "Distributed trainer"),
        ("light_malib.rollout.rollout_manager", "Rollout manager"),
        ("light_malib.rollout.rollout_worker", "Rollout worker"),
        ("light_malib.evaluation.evaluation_manager", "Evaluation manager"),
        ("light_malib.monitor.monitor", "Monitor"),
        ("light_malib.framework.pbt_runner", "PBT runner"),
        ("light_malib.agent.agent_manager", "Agent manager"),
        ("light_malib.registry.registry", "Registry"),
    ]
    
    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            print_status(name, True)
            results.add(True)
        except Exception as e:
            err_msg = str(e).split('\n')[0][:60]
            print_status(name, False, err_msg)
            results.add(False, name)
            all_ok = False
    
    return all_ok


def test_ray(results: TestResults, verbose=False):
    """Test Ray initialization."""
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(
            num_cpus=2, 
            num_gpus=0, 
            ignore_reinit_error=True, 
            log_to_driver=False,
            logging_level="ERROR"
        )
        resources = ray.cluster_resources()
        ray.shutdown()
        
        print_status("Ray init/shutdown", True, f"CPUs: {int(resources.get('CPU', 0))}")
        results.add(True)
        return True
    except Exception as e:
        print_status("Ray", False, str(e))
        results.add(False, "Ray")
        return False


def test_config_loading(results: TestResults, verbose=False):
    """Test loading training configs."""
    try:
        from light_malib.utils.cfg import load_cfg
        
        configs = [
            "expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/ippo.yaml",
        ]
        
        all_ok = True
        for config_path in configs:
            full_path = os.path.join(PROJECT_ROOT, config_path)
            if os.path.exists(full_path):
                try:
                    cfg = load_cfg(full_path)
                    print_status(f"Config: {os.path.basename(config_path)}", True)
                    results.add(True)
                except Exception as e:
                    print_status(f"Config: {os.path.basename(config_path)}", False, str(e))
                    results.add(False, f"Config {config_path}")
                    all_ok = False
            else:
                print_status(f"Config: {os.path.basename(config_path)}", False, "File not found")
                results.add(False, f"Config {config_path}")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print_status("Config loading", False, str(e))
        results.add(False, "Config loading")
        return False


def test_model_components(results: TestResults, verbose=False):
    """Test that model components can be instantiated."""
    all_ok = True
    
    try:
        from light_malib.algorithm.common.model import Model, MLP, RNN, get_model, mlp
        print_status("Model classes (Model, MLP, RNN)", True)
        results.add(True)
    except Exception as e:
        print_status("Model classes", False, str(e))
        results.add(False, "Model classes")
        all_ok = False
    
    try:
        from light_malib.registry import registry
        print_status("Model registry", True)
        results.add(True)
    except Exception as e:
        print_status("Model registry", False, str(e))
        results.add(False, "Model registry")
        all_ok = False
    
    try:
        from light_malib.algorithm.common.policy import Policy
        print_status("Base Policy class", True)
        results.add(True)
    except Exception as e:
        print_status("Base Policy class", False, str(e))
        results.add(False, "Base Policy class")
        all_ok = False
    
    return all_ok


def run_all(quick=False, gpu=False, verbose=False) -> TestResults:
    """Run all environment tests."""
    results = TestResults()
    
    print_subsection("Python Version")
    test_python_version(results)
    
    print_subsection("Core Dependencies")
    test_core_imports(results, verbose)
    
    if gpu:
        print_subsection("PyTorch & CUDA")
        test_pytorch_cuda(results, verbose)
    else:
        print("  [PyTorch & CUDA - skipped, use --gpu]")
    
    print_subsection("Ray Framework")
    test_ray(results, verbose)
    
    print_subsection("light_malib Imports")
    test_light_malib_imports(results, verbose)
    
    print_subsection("Model Components")
    test_model_components(results, verbose)
    
    if not quick:
        print_subsection("GFootball Environment")
        test_gfootball(results, verbose)
    else:
        print("  [GFootball Environment - skipped, remove --quick]")
    
    print_subsection("Config Loading")
    test_config_loading(results, verbose)
    
    return results

