// Step colours: look-alike steps share a hue family, so a confusion inside a pair reads as a shade change, and a
// confusion across families reads as a different colour.
export const STEP_COLOURS: Record<string, string> = {
  idle: "#c9cfcc",
  pick_housing: "#6e8b74",
  place_housing: "#4f6b57",
  pick_screw: "#8c7a5b",
  fasten_a: "#2f6f9e",
  fasten_b: "#7fb0d4",
  torque_a: "#173a55",
  insert_clip: "#c2702e",
  remove_clip: "#ebb47e",
  inspect: "#7d5a86",
  unload: "#3e3e3e",
};

export const STEP_WORDS: Record<string, string> = {
  idle: "hands at rest",
  pick_housing: "pick housing",
  place_housing: "place housing",
  pick_screw: "pick screw",
  fasten_a: "fasten at slot a",
  fasten_b: "fasten at slot b",
  torque_a: "torque check at a",
  insert_clip: "insert clip",
  remove_clip: "remove clip",
  inspect: "inspect",
  unload: "unload",
};

export const DURATION = { reveal: 0.6, change: 0.35, stagger: 0.08 } as const;
export const EASE = [0.2, 0.7, 0.2, 1] as const;

export const pct = (x: number | null | undefined, digits = 1): string =>
  x === null || x === undefined || Number.isNaN(x) ? "n/a" : `${(100 * x).toFixed(digits)}%`;

export const CONDITIONS = ["normal", "camera", "lighting", "busy", "variant"] as const;
export const ordered = (keys: string[]): string[] =>
  [...CONDITIONS.filter((c) => keys.includes(c)), ...keys.filter((k) => !(CONDITIONS as readonly string[]).includes(k))];
