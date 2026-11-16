import type { Report } from "../../shared/data/types";
import { pct } from "../../shared/tokens";
import { Section } from "../../shared/ui/Section";
import styles from "./edge.module.css";

export function EdgePanel({ report }: { report: Report }) {
  const rt = report.edge_runtime;
  const m = report.monitor;
  const flagged = m.rows.filter((r) => r.flagged || r.weak);
  return (
    <Section
      title="On the camera box"
      lede="Detector, crops, int8 encoder and temporal head in ONNX Runtime, one CPU thread, no PyTorch. A clip every ten seconds goes to the cloud model for a second opinion."
    >
      <dl className={styles.facts}>
        <div><dt>Per frame, int8</dt><dd className="tabular">{rt.int8.ms_per_frame} ms</dd></div>
        <div><dt>Per frame, fp32</dt><dd className="tabular">{rt.fp32.ms_per_frame} ms</dd></div>
        <div><dt>Same label as PyTorch</dt><dd className="tabular">{pct(rt.int8.agrees_with_pytorch)}</dd></div>
        <div><dt>Encoder, int8</dt><dd className="tabular">{rt.model_kb.encoder_int8} kB</dd></div>
        <div><dt>Weak recordings flagged</dt><dd className="tabular">{m.weak_flagged === null ? "none weak" : pct(m.weak_flagged)}</dd></div>
        <div><dt>Healthy recordings flagged</dt><dd className="tabular">{pct(m.healthy_flagged)}</dd></div>
      </dl>
      {flagged.length === 0 ? (
        <p className={styles.note}>No recording was weak and none was flagged.</p>
      ) : (
        <table className={styles.table}>
          <thead>
            <tr><th scope="col">Recording</th><th scope="col">Edge frames right</th><th scope="col">Weak</th><th scope="col">Flagged</th><th scope="col">Why</th></tr>
          </thead>
          <tbody>
            {flagged.map((r) => (
              <tr key={r.recording}>
                <th scope="row">{r.recording}</th>
                <td className="tabular">{pct(r.edge_accuracy)}</td>
                <td>{r.weak ? "yes" : "no"}</td>
                <td>{r.flagged ? `after ${r.flagged_after_s} s` : "no"}</td>
                <td>{r.why}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}
