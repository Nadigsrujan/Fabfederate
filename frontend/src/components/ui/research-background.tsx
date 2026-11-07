"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";
import { cn } from "@/lib/utils";

export interface ResearchBackgroundProps {
  className?: string;
  children?: ReactNode;
  intensity?: "subtle" | "medium" | "strong";
  animated?: boolean;
}

interface Beam {
  x: number; y: number; width: number; length: number; angle: number;
  speed: number; opacity: number; hue: number; pulse: number; pulseSpeed: number;
}

const opacityMap = { subtle: 0.7, medium: 0.85, strong: 1 };

function createBeam(width: number, height: number): Beam {
  return {
    x: Math.random() * width * 1.5 - width * 0.25,
    y: Math.random() * height * 1.5 - height * 0.25,
    width: 30 + Math.random() * 60, length: Math.max(height * 2.5, width),
    angle: -35 + Math.random() * 10, speed: 0.6 + Math.random() * 1.2,
    opacity: 0.12 + Math.random() * 0.16, hue: 190 + Math.random() * 70,
    pulse: Math.random() * Math.PI * 2, pulseSpeed: 0.02 + Math.random() * 0.03,
  };
}

/** Decorative research header with container sizing and reduced-motion support. */
export function ResearchBackground({ className, children, intensity = "strong", animated = true }: ResearchBackgroundProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const beamsRef = useRef<Beam[]>([]);
  const animationFrameRef = useRef<number>(0);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!host || !canvas || !ctx) return;
    let width = 1;
    let height = 1;
    let visible = true;
    let previousTime = 0;

    const drawBeam = (beam: Beam) => {
      ctx.save();
      ctx.translate(beam.x, beam.y);
      ctx.rotate((beam.angle * Math.PI) / 180);
      const opacity = beam.opacity * (0.8 + Math.sin(beam.pulse) * 0.2) * opacityMap[intensity];
      const gradient = ctx.createLinearGradient(0, 0, 0, beam.length);
      for (const [stop, strength] of [[0, 0], [0.1, 0.5], [0.4, 1], [0.6, 1], [0.9, 0.5], [1, 0]]) {
        gradient.addColorStop(stop, `hsla(${beam.hue}, 85%, 65%, ${opacity * strength})`);
      }
      ctx.fillStyle = gradient;
      ctx.fillRect(-beam.width / 2, 0, beam.width, beam.length);
      ctx.restore();
    };

    const paint = (elapsed = 0) => {
      ctx.clearRect(0, 0, width, height);
      ctx.filter = "blur(35px)";
      beamsRef.current.forEach((beam, index) => {
        beam.y -= beam.speed * elapsed;
        beam.pulse += beam.pulseSpeed * elapsed;
        if (beam.y + beam.length < -100) {
          const spacing = width / 3;
          beam.y = height + 100;
          beam.x = (index % 3) * spacing + spacing / 2 + (Math.random() - 0.5) * spacing * 0.5;
          beam.width = 100 + Math.random() * 100;
          beam.speed = 0.5 + Math.random() * 0.4;
          beam.hue = 190 + (index * 70) / beamsRef.current.length;
          beam.opacity = 0.2 + Math.random() * 0.1;
        }
        drawBeam(beam);
      });
    };

    const stop = () => {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = 0;
      previousTime = 0;
    };
    const animate = (time: number) => {
      const elapsed = previousTime ? Math.min(time - previousTime, 32) / (1000 / 60) : 1;
      previousTime = time;
      paint(elapsed);
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    const resume = () => {
      stop();
      if (animated && !reducedMotion && visible && !document.hidden) {
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    };
    const updateCanvasSize = () => {
      const bounds = host.getBoundingClientRect();
      width = Math.max(1, bounds.width);
      height = Math.max(1, bounds.height);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      beamsRef.current = Array.from({ length: width < 640 ? 18 : 30 }, () => createBeam(width, height));
      paint();
    };

    updateCanvasSize();
    resume();
    const resizeObserver = new ResizeObserver(updateCanvasSize);
    resizeObserver.observe(host);
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      resume();
    });
    intersectionObserver.observe(host);
    window.addEventListener("resize", updateCanvasSize);
    document.addEventListener("visibilitychange", resume);
    return () => {
      stop();
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      window.removeEventListener("resize", updateCanvasSize);
      document.removeEventListener("visibilitychange", resume);
    };
  }, [animated, intensity, reducedMotion]);

  return (
    <div ref={hostRef} className={cn("relative min-h-screen w-full overflow-hidden bg-neutral-950", className)}>
      <canvas ref={canvasRef} aria-hidden="true" className="pointer-events-none absolute inset-0" style={{ filter: "blur(15px)" }} />
      {animated && <motion.div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-neutral-950/5"
        animate={reducedMotion ? { opacity: 0.05 } : { opacity: [0.05, 0.15, 0.05] }}
        transition={{ duration: 10, ease: "easeInOut", repeat: reducedMotion ? 0 : Infinity }}
        style={{ backdropFilter: "blur(50px)" }} />}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
