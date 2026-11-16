import { motion } from "motion/react";
import type { Report } from "../../shared/data/types";
import { DURATION, EASE, STEP_WORDS, pct } from "../../shared/tokens";
import { Bar } from "../../shared/ui/Bar";
import styles from "./hero.module.css";

const TEACHER = "vlm_lora_pairs_crops";

export function Hero({ report }: { report: Report }) {
  const n = (k: string) => report.vlm[k]?.conditions.normal.lookalike_accuracy ?? NaN;
  const rows: [string, number, "ink" | "soft"][] = [
    ["Fine-tuned with LoRA, the look-alike loss and hand crops", n(TEACHER), "ink"],
    ["LoRA alone", n("vlm_lora"), "soft"],
    ["Base model, no labels from this station", n("vlm_zero_shot"), "soft"],
  ];
  const pair = ["fasten_a", "fasten_b"].filter((s) => report.showcase.frames[s]);
  const item = (i: number) => ({
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: DURATION.reveal, ease: EASE, delay: i * DURATION.stagger },
  });
  return (
    <header className={styles.hero}>
      <motion.h1 className={styles.title} {...item(0)}>
        Slot a or slot b?
      </motion.h1>
      <motion.p className={styles.lede} {...item(1)}>
        On station {report.world.customer} the two screw slots sit {report.world.slot_gap.toFixed(0)} units apart on a 96-unit bench.
        A camera sees a few pixels of difference, and the hand crops beside each frame are where the model looks for them.
      </motion.p>
      <div className={styles.pair}>
        {pair.map((s, i) => (
          <motion.figure key={s} className={styles.figure} {...item(2 + i)}>
            <img src={`data:image/png;base64,${report.showcase.frames[s]}`} alt={`A frame of '${STEP_WORDS[s]}' with both hand crops`} />
            <figcaption>{STEP_WORDS[s]}</figcaption>
          </motion.figure>
        ))}
      </div>
      <motion.div className={styles.results} {...item(4)}>
        <p className={styles.question}>Frames of look-alike steps labelled right</p>
        {rows.map(([label, v, tone]) => (
          <div className={styles.row} key={label}>
            <span className={styles.label}>{label}</span>
            <span className={`${styles.value} tabular`}>{pct(v)}</span>
            <Bar value={v} tone={tone} label={`${label}: ${pct(v)}`} />
          </div>
        ))}
      </motion.div>
    </header>
  );
}
