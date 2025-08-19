


from prereq_checker import (
    check_wsl,
    check_wsl_distro,
    check_docker,
    check_docker_wsl_integration,
    check_python,
    check_gpu_drivers,
)

def run_checks():
    checks = [
        check_wsl(),
        check_wsl_distro(),
        check_docker(),
        check_docker_wsl_integration(),
        check_python(),
        check_gpu_drivers(),
    ]
    results = ""
    all_passed = True
    for success, message in checks:
        results += f"{'✅' if success else '❌'} {message}\n"
        if not success:
            all_passed = False
    return results, all_passed



