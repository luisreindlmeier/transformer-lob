def banner(text: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def get_model_name(base: str, decay: bool = False) -> str:
    if base.upper().replace("-", "").replace("_", "") == "MAJORITY":
        return "Majority"
    return f"{base}-Decay" if decay else base
