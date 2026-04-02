"""Simple governance risk helper using the V-Dem liberal democracy score."""

def calculate_governance_risk(vdem_score: float) -> tuple[float, str]:
    """Convert a V-Dem Liberal Democracy score into a 0-100 risk metric.

    The Liberal Democracy index (v2x_libdem) ranges from 0 (least democratic) to 1 (most democratic).
    Risk is inverted so that stronger liberal democracy means lower risk, using the formula
        risk = (1 - vdem_score) * 100

    Returns the numeric risk score and a qualitative classification:
        Low (<30), Medium (30-60), High (>60).
    """
    if not 0.0 <= vdem_score <= 1.0:
        raise ValueError("vdem_score must be between 0 and 1")

    # Invert the democracy score so that more autocratic regimes score higher risk.
    risk_score = (1.0 - vdem_score) * 100.0

    if risk_score < 30.0:
        classification = "Low"
    elif risk_score <= 60.0:
        classification = "Medium"
    else:
        classification = "High"

    return risk_score, classification


if __name__ == "__main__":
    sample_country = "Freedonia"
    sample_vdem = 0.42  # Example V-Dem liberal democracy index for the country.
    score, label = calculate_governance_risk(sample_vdem)
    print(
        f"{sample_country}: v2x_libdem={sample_vdem} -> governance risk {score:.1f} ({label})"
    )
