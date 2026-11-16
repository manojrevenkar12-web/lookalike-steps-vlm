import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";
import type { Report } from "../../shared/data/types";
import { DURATION, EASE, STEP_COLOURS, STEP_WORDS } from "../../shared/tokens";
import { Section } from "../../shared/ui/Section";
import styles from "./timeline.module.css";

type Run = { label: number; from: number; to: number };
const runs = (xs: number[]): Run[] => {
  const out: Run[] = [];
  xs.forEach((x, i) => {
    const last = out[out.length - 1];
    if (last && last.label === x) last.to = i;
    else out.push({ label: x, from: i, to: i });
  });
  return out;
};

const CONDITION_WORDS: Record<string, string> = { normal: "As recorded for training", camera: "Camera moved" };

export function Timeline({ report }: { report: Report }) {
  const recs = report.showcase.recordings;
  const [which, setWhich] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const rec = recs[which];
  const n = rec?.truth.length ?? 0;
  const rows = useMemo(
    () =>
      rec
        ? [
            { name: "What happened", data: rec.truth },
            { name: "Cloud model", data: rec.cloud },
            { name: "Edge model", data: rec.edge },
            { name: "Edge reports", data: rec.edge.map((x, i) => (rec.reported[i] ? x : -1)) },
          ]
        : [],
    [rec],
  );
  if (!rec) return null;
  const at = hover ?? null;
  return (
    <Section
      title="One minute, frame by frame"
      lede="Each band is a step. Look-alike steps share a colour family, so a slot mix-up shows as a change of shade. Grey gaps in the last band are frames the edge held back."
    >
      <div className={styles.switch} role="group" aria-label="Recording">
        {recs.map((r, i) => (
          <button key={r.key} aria-pressed={i === which} onClick={() => setWhich(i)} className={styles.option}>
            {CONDITION_WORDS[r.condition] ?? r.condition}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div
          key={rec.key}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: DURATION.change, ease: EASE }}
          className={styles.chart}
          onMouseLeave={() => setHover(null)}
        >
          {rows.map((row) => (
            <div className={styles.row} key={row.name}>
              <span className={styles.name}>{row.name}</span>
              <svg
                className={styles.ribbon}
                viewBox={`0 0 ${n} 1`}
                preserveAspectRatio="none"
                role="img"
                aria-label={`${row.name}: ${runs(row.data).length} segments`}
                onMouseMove={(e) => {
                  const box = e.currentTarget.getBoundingClientRect();
                  setHover(Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * n))));
                }}
              >
                {runs(row.data).map((r) => (
                  <rect key={r.from} x={r.from} y={0} width={r.to - r.from + 1} height={1}
                    fill={r.label < 0 ? "url(#held)" : STEP_COLOURS[report.steps[r.label]]} />
                ))}
                {at !== null && <rect x={at} y={0} width={0.6} height={1} className={styles.cursor} />}
              </svg>
            </div>
          ))}
          <svg width="0" height="0" aria-hidden="true">
            <defs>
              <pattern id="held" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="4" height="4" fill="var(--paper)" />
                <rect width="1.5" height="4" fill="var(--held)" />
              </pattern>
            </defs>
          </svg>
        </motion.div>
      </AnimatePresence>
      <p className={`${styles.readout} tabular`} aria-live="polite">
        {at === null
          ? "Point at the bands to read a frame."
          : `${(at / 5).toFixed(1)} s · happened: ${STEP_WORDS[report.steps[rec.truth[at]]]} · cloud: ${STEP_WORDS[report.steps[rec.cloud[at]]]} · edge: ${
              rec.reported[at] ? STEP_WORDS[report.steps[rec.edge[at]]] : "held back"
            }`}
      </p>
      <ul className={styles.legend}>
        {report.world.standard_work
          .filter((s, i, a) => a.indexOf(s) === i)
          .concat("idle")
          .map((s) => (
            <li key={s}>
              <span className={styles.swatch} style={{ background: STEP_COLOURS[s] }} />
              {STEP_WORDS[s]}
            </li>
          ))}
      </ul>
    </Section>
  );
}
