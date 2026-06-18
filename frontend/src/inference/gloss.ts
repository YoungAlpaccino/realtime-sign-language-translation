// Demo gloss vocab: until a trained vocab/checkpoint exists, map CTC ids to
// readable placeholder tokens so the caption is human-visible.
export function idsToGloss(ids: number[]): string {
  return ids.map((i) => `G${i}`).join(" ");
}

/** Ask the backend SLT endpoint to render gloss -> natural-language text. */
export async function translate(gloss: string): Promise<string> {
  if (!gloss.trim()) return "";
  const res = await fetch("/api/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gloss }),
  });
  if (!res.ok) return "";
  const data = (await res.json()) as { text: string };
  return data.text;
}
