import type { Report } from "../../shared/data/types";
import { ordered, pct } from "../../shared/tokens";
import { Bar } from "../../shared/ui/Bar";
import { Section } from "../../shared/ui/Section";
import styles from "./conditions.module.css";

const WORDS: Record<string, string> = {
  normal: "As recorded for training",
  camera: "Camera moved",
  lighting: "Light and noise",
  busy: "People walking past",
  variant: "New product variant",
};

export function Conditions({ report }: { report: Report }) {
  const conds = ordered(Object.keys(report.reported));
  return (
    <Section
      title="When the station changes"
      lede="The cloud and edge models under conditions they were not trained on, and what the edge still reports. The gate trades coverage for accuracy: it reports fewer frames when it is unsure."
    >
      <div className={styles.grid}>
        {conds.map((c) => {
          const cloud = report.vlm.vlm_lora_pairs_crops.conditions[c];
          const edge = report.edge.edge_student.conditions[c];
          const rep = report.reported[c];
          const items: [string, number, "ink" | "soft"][] = [
            ["Cloud: frames right", cloud.frame_accuracy, "soft"],
            ["Edge: frames right", edge.frame_accuracy, "soft"],
            ["Edge: frames reported", rep.coverage, "soft"],
            ["Edge: reported frames right", rep.reported_accuracy, "ink"],
          ];
          return (
            <article key={c} className={styles.card}>
              <h3 className={styles.name}>{WORDS[c] ?? c}</h3>
              {items.map(([label, v, tone]) => (
                <div key={label} className={styles.item}>
                  <span>{label}</span>
                  <span className="tabular">{pct(v)}</span>
                  <Bar value={v} tone={tone} label={`${label} ${pct(v)}`} />
                </div>
              ))}
            </article>
          );
        })}
      </div>
    </Section>
  );
}
