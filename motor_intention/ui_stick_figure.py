from __future__ import annotations

from html import escape


def _segment_style(active: bool, phase_name: str, family: str | None) -> tuple[str, int, float]:
    base_color = "#5A6C82"
    if family == "arm":
        base_color = "#FF7A59"
    elif family == "leg":
        base_color = "#2EC4B6"

    intensity = {
        "baseline": 0.20,
        "preparation": 0.55,
        "imagery": 1.0,
        "recovery": 0.45,
    }.get(phase_name, 0.3)

    if not active:
        return "#8FA7BF", 8, 0.35

    width = int(9 + 5 * intensity)
    opacity = 0.65 + 0.3 * intensity
    return base_color, width, opacity


def render_stick_figure_svg(
    flat_label: str,
    effector_family: str,
    side: str | None,
    phase_name: str,
    emphasis_family: str | None = None,
    emphasis_side: str | None = None,
) -> str:
    active_family = emphasis_family if emphasis_family in {"arm", "leg"} else effector_family
    active_side = emphasis_side if emphasis_side in {"left", "right"} else side

    left_arm_active = active_family == "arm" and active_side == "left"
    right_arm_active = active_family == "arm" and active_side == "right"
    left_leg_active = active_family == "leg" and active_side == "left"
    right_leg_active = active_family == "leg" and active_side == "right"

    left_arm = _segment_style(left_arm_active, phase_name, "arm" if left_arm_active else None)
    right_arm = _segment_style(right_arm_active, phase_name, "arm" if right_arm_active else None)
    left_leg = _segment_style(left_leg_active, phase_name, "leg" if left_leg_active else None)
    right_leg = _segment_style(right_leg_active, phase_name, "leg" if right_leg_active else None)

    torso_glow = "#2DD4FF" if phase_name in {"preparation", "imagery"} else "#18314F"
    head_fill = "#102341" if flat_label != "rest" else "#0B1730"

    return f"""
    <div style="background:radial-gradient(circle at top,#102341 0%,#071122 68%);border:1px solid #18314F;border-radius:20px;padding:10px 6px;box-shadow:inset 0 0 0 1px rgba(45,212,255,0.05);">
      <svg viewBox="0 0 280 380" width="100%" height="380" role="img" aria-label="Motor intention stick figure">
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <circle cx="140" cy="50" r="28" fill="{head_fill}" stroke="#8FA7BF" stroke-width="3"/>
        <path d="M 112 92 Q 140 82 168 92 L 160 182 Q 140 196 120 182 Z" fill="#0B1730" stroke="{torso_glow}" stroke-width="3" opacity="0.95" filter="url(#glow)"/>
        <line x1="140" y1="94" x2="140" y2="186" stroke="#8FA7BF" stroke-width="8" stroke-linecap="round" opacity="0.85"/>
        <line x1="140" y1="104" x2="88" y2="138" stroke="{left_arm[0]}" stroke-width="{left_arm[1]}" stroke-opacity="{left_arm[2]}" stroke-linecap="round" filter="url(#glow)"/>
        <line x1="88" y1="138" x2="62" y2="194" stroke="{left_arm[0]}" stroke-width="{left_arm[1]}" stroke-opacity="{left_arm[2]}" stroke-linecap="round" filter="url(#glow)"/>
        <line x1="140" y1="104" x2="192" y2="138" stroke="{right_arm[0]}" stroke-width="{right_arm[1]}" stroke-opacity="{right_arm[2]}" stroke-linecap="round" filter="url(#glow)"/>
        <line x1="192" y1="138" x2="218" y2="194" stroke="{right_arm[0]}" stroke-width="{right_arm[1]}" stroke-opacity="{right_arm[2]}" stroke-linecap="round" filter="url(#glow)"/>
        <line x1="132" y1="186" x2="102" y2="258" stroke="{left_leg[0]}" stroke-width="{left_leg[1]}" stroke-opacity="{left_leg[2]}" stroke-linecap="round" filter="url(#glow)"/>
        <line x1="102" y1="258" x2="92" y2="344" stroke="{left_leg[0]}" stroke-width="{left_leg[1]}" stroke-opacity="{left_leg[2]}" stroke-linecap="round" filter="url(#glow)"/>
        <line x1="148" y1="186" x2="178" y2="258" stroke="{right_leg[0]}" stroke-width="{right_leg[1]}" stroke-opacity="{right_leg[2]}" stroke-linecap="round" filter="url(#glow)"/>
        <line x1="178" y1="258" x2="188" y2="344" stroke="{right_leg[0]}" stroke-width="{right_leg[1]}" stroke-opacity="{right_leg[2]}" stroke-linecap="round" filter="url(#glow)"/>
        <circle cx="140" cy="104" r="7" fill="#8FA7BF"/>
        <circle cx="132" cy="186" r="6" fill="#8FA7BF"/>
        <circle cx="148" cy="186" r="6" fill="#8FA7BF"/>
        <circle cx="88" cy="138" r="6" fill="#8FA7BF"/>
        <circle cx="192" cy="138" r="6" fill="#8FA7BF"/>
        <circle cx="102" cy="258" r="6" fill="#8FA7BF"/>
        <circle cx="178" cy="258" r="6" fill="#8FA7BF"/>
        <text x="140" y="20" text-anchor="middle" fill="#D6E3F0" font-size="16" font-weight="700">{escape(flat_label.replace("_", " ").title())}</text>
        <text x="140" y="368" text-anchor="middle" fill="#8FA7BF" font-size="13">Phase: {escape(phase_name.title())}</text>
      </svg>
    </div>
    """
