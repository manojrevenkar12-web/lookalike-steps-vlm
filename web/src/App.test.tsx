import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const report = {
  label: "dev",
  steps: ["idle", "fasten_a", "fasten_b"],
  lookalike: [["fasten_a", "fasten_b"]],
  world: { customer: "C1", slot_gap: 6, labelled_recordings: 4, seconds_per_recording: 60, standard_work: ["fasten_a", "fasten_b"] },
  base: { params: 1000 },
  vlm: Object.fromEntries(
    ["vlm_zero_shot", "vlm_lora", "vlm_lora_pairs_crops"].map((k) => [
      k,
      { trainable_params: 10, conditions: { normal: { frame_accuracy: 0.9, lookalike_accuracy: 0.8, edit: 90, f1_50: 90, grounding_r50: 0.8, grounding_miou: 0.7 } } },
    ]),
  ),
  edge: { edge_student: { params: 5, conditions: { normal: { frame_accuracy: 0.88, lookalike_accuracy: 0.8, edit: 80, f1_50: 80, grounding_r50: 0.7, grounding_miou: 0.6 } } } },
  agreement_edge_cloud: { normal: 0.9 },
  tracking: { normal: { side: { idf1: 0.9, id_switches_per_min: 0.5, recall: 0.9 }, overlap: { idf1: 0.3, id_switches_per_min: 8, recall: 0.6 } } },
  reported: { normal: { reported_accuracy: 0.97, coverage: 0.8 } },
  monitor: { weak_recordings: 0, weak_flagged: null, healthy_recordings: 1, healthy_flagged: 0, rows: [] },
  edge_runtime: { int8: { ms_per_frame: 3, agrees_with_pytorch: 1 }, fp32: { ms_per_frame: 2 }, model_kb: { encoder_int8: 400 } },
  curve: {},
  showcase: { recordings: [{ key: "C1/normal/0", condition: "normal", truth: [0, 1, 2], cloud: [0, 1, 1], edge: [0, 1, 2], reported: [1, 0, 1] }], frames: {} },
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("shows a skeleton while loading, then the results", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(report), { status: 200 }));
    render(<App />);
    expect(screen.getByLabelText("Loading the results")).toBeTruthy();
    await waitFor(() => expect(screen.getByText("Slot a or slot b?")).toBeTruthy());
    expect(screen.getByText("What each change buys")).toBeTruthy();
  });

  it("explains an error and loads again on request", async () => {
    const f = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response("", { status: 500 }));
    f.mockResolvedValueOnce(new Response(JSON.stringify(report), { status: 200 }));
    render(<App />);
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    fireEvent.click(screen.getByText("Load again"));
    await waitFor(() => expect(screen.getByText("Slot a or slot b?")).toBeTruthy());
  });

  it("says what to run when there is no report", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("", { status: 404 }));
    render(<App />);
    await waitFor(() => expect(screen.getByText("No results yet.")).toBeTruthy());
  });
});
