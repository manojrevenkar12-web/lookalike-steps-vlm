import type { Report } from "../../shared/data/types";
import { ordered } from "../../shared/tokens";
import { Section } from "../../shared/ui/Section";
import styles from "./hands.module.css";

export function Hands({ report }: { report: Report }) {
  const conds = ordered(Object.keys(report.tracking));
  return (
    <Section
      title="Which hand is which"
      lede="The crops only help if they follow the right hand. A DETR that tells the left hand from the right by its arm keeps identities where overlap tracking hands them to the other hand at the fixture."
    >
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Condition</th>
            <th scope="col">Identity F1, left/right DETR</th>
            <th scope="col">Identity F1, overlap tracking</th>
            <th scope="col">Switches a minute, left/right · overlap</th>
          </tr>
        </thead>
        <tbody>
          {conds.map((c) => {
            const t = report.tracking[c];
            return (
              <tr key={c}>
                <th scope="row">{c}</th>
                <td className="tabular">{t.side.idf1.toFixed(2)}</td>
                <td className="tabular">{t.overlap.idf1.toFixed(2)}</td>
                <td className="tabular">
                  {t.side.id_switches_per_min} · {t.overlap.id_switches_per_min}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Section>
  );
}
