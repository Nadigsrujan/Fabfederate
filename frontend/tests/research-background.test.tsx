import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ResearchBackground } from "@/components/ui/research-background";

let resize: () => void;
const setTransform = vi.fn();
const cancelFrame = vi.fn();

beforeEach(() => {
  vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: true, addEventListener: vi.fn(),
    removeEventListener: vi.fn(), addListener: vi.fn(), removeListener: vi.fn() })));
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: () => void) { resize = callback; }
    observe() {} disconnect() {}
  });
  vi.stubGlobal("IntersectionObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("requestAnimationFrame", vi.fn(() => 7));
  vi.stubGlobal("cancelAnimationFrame", cancelFrame);
  Object.defineProperty(window, "devicePixelRatio", { configurable: true, value: 2 });
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
    width: 640, height: 320, x: 0, y: 0, top: 0, left: 0, bottom: 320, right: 640,
    toJSON: () => ({}),
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    setTransform, clearRect: vi.fn(), save: vi.fn(), restore: vi.fn(), translate: vi.fn(),
    rotate: vi.fn(), fillRect: vi.fn(), createLinearGradient: () => ({ addColorStop: vi.fn() }),
  } as unknown as CanvasRenderingContext2D);
});

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("ResearchBackground", () => {
  it("renders application children instead of hard-coded demo copy", () => {
    render(<ResearchBackground><h1>Milling evidence</h1></ResearchBackground>);
    expect(screen.getByRole("heading", { name: "Milling evidence" })).toBeVisible();
    expect(screen.queryByText("For your pleasure")).not.toBeInTheDocument();
  });

  it("sizes its backing store at DPR while drawing in container CSS coordinates", () => {
    const { container } = render(<ResearchBackground intensity="subtle" />);
    const canvas = container.querySelector("canvas")!;
    expect(canvas.width).toBe(1280);
    expect(canvas.height).toBe(640);
    expect(setTransform).toHaveBeenCalledWith(2, 0, 0, 2, 0, 0);
    resize();
    expect(canvas.width).toBe(1280);
  });

  it("keeps decoration inaccessible and does not loop under reduced motion", () => {
    const { container, unmount } = render(<ResearchBackground />);
    expect(container.querySelector("canvas")).toHaveAttribute("aria-hidden", "true");
    expect(requestAnimationFrame).not.toHaveBeenCalled();
    fireEvent(window, new Event("resize"));
    unmount();
  });
});
