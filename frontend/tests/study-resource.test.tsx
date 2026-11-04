import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useStudy } from "@/lib/experiments";

afterEach(() => vi.unstubAllGlobals());

it("never presents the previous experiment's data under a newly selected identifier", async () => {
  const response = (body: unknown) => new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    if (url === "/api/experiments") return Promise.resolve(response({
      experiments: [{ id: "first" }, { id: "second" }], default_id: "first",
    }));
    if (url.endsWith("/first")) return Promise.resolve(response({ id: "first" }));
    return new Promise<Response>(() => {});
  }));
  const renders: [string, string | undefined][] = [];
  const { result, unmount } = renderHook(() => {
    const resource = useStudy(); renders.push([resource.id, resource.study?.id]); return resource;
  });
  await waitFor(() => expect(result.current.study?.id).toBe("first"));
  act(() => result.current.setId("second"));
  expect(renders.some(([selected, displayed]) => selected === "second" && displayed === "first")).toBe(false);
  expect(result.current.loading).toBe(true);
  unmount();
});
