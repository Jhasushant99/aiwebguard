def decide(risk: str) -> str:
    if risk == "HIGH":
        return " DANGEROUS"
    elif risk == "MEDIUM":
        return " SUSPICIOUS"
    return " SAFE"