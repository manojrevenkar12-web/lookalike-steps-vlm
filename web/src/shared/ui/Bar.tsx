import { motion } from "motion/react";
import { DURATION, EASE } from "../tokens";
import styles from "./bar.module.css";

export function Bar({ value, tone = "ink", label }: { value: number; tone?: "ink" | "soft" | "wrong"; label: string }) {
  return (
    <div className={styles.track} role="img" aria-label={label}>
      <motion.div
        className={`${styles.fill} ${styles[tone]}`}
        initial={{ scaleX: 0 }}
        animate={{ scaleX: Math.max(0, Math.min(1, value)) }}
        transition={{ duration: DURATION.reveal, ease: EASE }}
      />
    </div>
  );
}
