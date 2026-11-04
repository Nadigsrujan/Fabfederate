import type { ReactNode, SelectHTMLAttributes } from "react";
import { ChevronDown, CircleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Select({ label, children, ...props }: SelectHTMLAttributes<HTMLSelectElement> & { label: string }) {
  return <label className="field-select"><span>{label}</span><div><select {...props}>{children}</select><ChevronDown size={15} aria-hidden="true" /></div></label>;
}
export function SectionHeading({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return <div className="section-heading"><div><h2>{title}</h2>{description && <p>{description}</p>}</div>{action}</div>;
}
export function InfoNote({ children }: { children: ReactNode }) {
  return <div className="info-note"><CircleAlert size={16} aria-hidden="true" /><p>{children}</p></div>;
}
export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return <section className="error-state" role="alert"><CircleAlert size={28} /><h2>Research data unavailable</h2><p>{message}</p><p>Check that the local API is running, then reconnect.</p>{retry && <Button onClick={retry}>Reconnect</Button>}<details><summary>Local setup</summary><code>.venv/bin/python -m uvicorn fabfederate.api.app:app --host 127.0.0.1 --port 8000</code></details></section>;
}
