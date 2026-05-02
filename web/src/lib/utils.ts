import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/** The shadcn standard `cn` helper — merges class lists, dedupes Tailwind. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
