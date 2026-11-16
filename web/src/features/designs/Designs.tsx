import type { Report } from "../../shared/data/types";
import { pct } from "../../shared/tokens";
import { Bar } from "../../shared/ui/Bar";
import { Section } from "../../shared/ui/Section";
import styles from "./designs.module.css";

const WORDS: Record<string, string> = {
  vlm_zero_shot: "Base model, no customer labels",
  vlm_full: "Full fine-tune",
  vlm_lora: "LoRA",
  vlm_lora_pairs: "LoRA + look-alike loss",
  vlm_lora_pairs_crops: "LoRA + look-alike loss + hand crops",
  scratch_student: "Edge model from labels only",
  edge_student: "Edge model distilled from the cloud model",
};
const PRODUCT = new Set(["vlm_lora_pairs_crops", "edge_student"]);

export function Designs({ report }: { report: Report }) {
  const order = Object.keys(WORDS);
  const rank = (k: string) => (order.indexOf(k) < 0 ? order.length : order.indexOf(k));
  const rows = [
    ...Object.entries(report.vlm).map(([k, v]) => ({ k, s: v.conditions.normal, where: "cloud" })),
    ...Object.entries(report.edge).map(([k, v]) => ({ k, s: v.conditions.normal, where: "edge" })),
  ].sort((a, b) => rank(a.k) - rank(b.k));
  return (
    <Section
      title="What each change buys"
      lede={`Every design is trained on the same ${report.world.labelled_recordings} labelled recordings and tested on the same new ones. Each line changes one thing.`}
    >
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Design</th>
            <th scope="col">Runs on</th>
            <th scope="col">Frames right</th>
            <th scope="col">Look-alike steps right</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ k, s, where }) => (
            <tr key={k} className={PRODUCT.has(k) ? styles.product : undefined}>
              <th scope="row">{WORDS[k] ?? k}</th>
              <td>{where}</td>
              <td>
                <span className="tabular">{pct(s.frame_accuracy)}</span>
                <Bar value={s.frame_accuracy} tone={PRODUCT.has(k) ? "ink" : "soft"} label={`frames right ${pct(s.frame_accuracy)}`} />
              </td>
              <td>
                <span className="tabular">{pct(s.lookalike_accuracy)}</span>
                <Bar value={s.lookalike_accuracy} tone={PRODUCT.has(k) ? "ink" : "soft"} label={`look-alike right ${pct(s.lookalike_accuracy)}`} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}
