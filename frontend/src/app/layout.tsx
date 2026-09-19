import type { Metadata } from "next";
import { Fraunces, Source_Sans_3 } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-sans",
});

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
});

export const metadata: Metadata = {
  title: "Smash Lab — Badminton Technique Analysis",
  description:
    "Upload a smash, review stroke phases, evidence-backed findings, and coaching focus.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${sourceSans.variable} ${fraunces.variable} antialiased`}
      >
        <div className="min-h-screen">
          <header className="border-b border-[var(--border)] bg-[var(--bg-elevated)]">
            <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
              <Link
                href="/"
                className="font-display text-xl font-semibold tracking-tight text-[var(--fg)]"
              >
                Smash Lab
              </Link>
              <nav
                className="flex items-center gap-1 text-sm font-medium text-[var(--muted)]"
                aria-label="Primary"
              >
                <Link
                  href="/analyze"
                  className="rounded-md px-3 py-2 hover:bg-[var(--accent-soft)] hover:text-[var(--accent)]"
                >
                  Analyze
                </Link>
                <Link
                  href="/history"
                  className="rounded-md px-3 py-2 hover:bg-[var(--accent-soft)] hover:text-[var(--accent)]"
                >
                  History
                </Link>
                <Link
                  href="/compare"
                  className="rounded-md px-3 py-2 hover:bg-[var(--accent-soft)] hover:text-[var(--accent)]"
                >
                  Compare
                </Link>
              </nav>
            </div>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
