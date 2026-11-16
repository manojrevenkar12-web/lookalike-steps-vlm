export type Scores = {
  frame_accuracy: number;
  lookalike_accuracy: number;
  edit: number;
  f1_50: number;
  grounding_r50: number;
  grounding_miou: number;
};
export type Recording = { key: string; condition: string; truth: number[]; cloud: number[]; edge: number[]; reported: number[] };
export type MonitorRow = { recording: string; condition: string; edge_accuracy: number; weak: boolean; flagged: boolean; flagged_after_s: number | null; why: string };
export type Report = {
  label: string;
  steps: string[];
  lookalike: [string, string][];
  world: { customer: string; slot_gap: number; labelled_recordings: number; seconds_per_recording: number; standard_work: string[] };
  base: { params: number };
  vlm: Record<string, { trainable_params: number; conditions: Record<string, Scores> }>;
  edge: Record<string, { params: number; conditions: Record<string, Scores> }>;
  agreement_edge_cloud: Record<string, number>;
  tracking: Record<string, Record<"side" | "overlap", { idf1: number; id_switches_per_min: number; recall: number }>>;
  reported: Record<string, { reported_accuracy: number; coverage: number }>;
  monitor: { weak_recordings: number; weak_flagged: number | null; healthy_recordings: number; healthy_flagged: number; rows: MonitorRow[] };
  edge_runtime: { int8: { ms_per_frame: number; agrees_with_pytorch: number }; fp32: { ms_per_frame: number }; model_kb: Record<string, number> };
  curve: Record<string, Scores>;
  showcase: { recordings: Recording[]; frames: Record<string, string> };
};
