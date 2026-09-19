"use client";

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

export type VideoPlayerHandle = {
  seekTo: (seconds: number, opts?: { pause?: boolean }) => void;
  getCurrentTime: () => number;
};

type Props = {
  src: string | null;
  unavailableMessage?: string;
  onTimeUpdate?: (t: number) => void;
  className?: string;
};

export const VideoPlayer = forwardRef<VideoPlayerHandle, Props>(
  function VideoPlayer(
    {
      src,
      unavailableMessage = "Analysis video is unavailable for this session.",
      onTimeUpdate,
      className,
    },
    ref
  ) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [error, setError] = useState(false);

    useImperativeHandle(ref, () => ({
      seekTo(seconds, opts) {
        const el = videoRef.current;
        if (!el || !Number.isFinite(seconds)) return;
        el.currentTime = Math.max(0, seconds);
        if (opts?.pause) {
          void el.pause();
        }
      },
      getCurrentTime() {
        return videoRef.current?.currentTime ?? 0;
      },
    }));

    useEffect(() => {
      setError(false);
    }, [src]);

    if (!src || error) {
      return (
        <div
          className="flex aspect-video w-full items-center justify-center rounded-lg border border-[var(--border)] bg-[#111] px-6 text-center text-sm text-white/80"
          role="status"
        >
          {unavailableMessage}
        </div>
      );
    }

    return (
      <video
        ref={videoRef}
        src={src}
        controls
        playsInline
        className={
          className ||
          "aspect-video w-full rounded-lg border border-[var(--border)] bg-black"
        }
        onTimeUpdate={(e) => onTimeUpdate?.(e.currentTarget.currentTime)}
        onError={() => setError(true)}
      />
    );
  }
);
