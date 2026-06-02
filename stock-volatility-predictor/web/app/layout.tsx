import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Volatility Watch | NASDAQ Risk Monitor",
  description: "Five-day NASDAQ volatility risk predictions powered by LightGBM",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
