import { writable, derived, get } from 'svelte/store';
import type { Job } from '$lib/api';
import { api } from '$lib/api';

/** First running job (backward compat for activity bar). */
export const currentJob = writable<Job | null>(null);
/** All currently running jobs. */
export const runningJobs = writable<Job[]>([]);
export const queuedJobs = writable<Job[]>([]);
export const jobHistory = writable<Job[]>([]);

/** Timestamp when the current job started (for ETA calculation). */
export const jobStartedAt = writable<number | null>(null);

export const activeJobCount = derived(
	[runningJobs, queuedJobs],
	([$running, $queued]) => $running.length + $queued.length
);

let refreshPending = false;

export async function refreshJobs() {
	if (refreshPending) return;
	refreshPending = true;
	try {
		const res = await api.jobs.list();
		const prev = get(currentJob);
		currentJob.set(res.current);
		runningJobs.set(res.running ?? (res.current ? [res.current] : []));
		queuedJobs.set(res.queued);
		jobHistory.set(res.history);

		// Track when a new job starts running
		if (res.current && (!prev || prev.id !== res.current.id)) {
			jobStartedAt.set(Date.now());
		} else if (!res.current) {
			jobStartedAt.set(null);
		}
	} catch {
		// silently fail
	} finally {
		refreshPending = false;
	}
}

/**
 * Update a job from a WebSocket message.
 * Returns true if the job was found and updated, false if not.
 */
export function updateJobFromWS(jobId: string, updates: Partial<Job>): boolean {
	let found = false;

	// Check running jobs
	runningJobs.update((jobs) =>
		jobs.map((j) => {
			if (j.id === jobId) {
				found = true;
				return { ...j, ...updates };
			}
			return j;
		})
	);

	// Also update currentJob for backward compat
	currentJob.update((j) => {
		if (j && j.id === jobId) {
			found = true;
			return { ...j, ...updates };
		}
		return j;
	});

	if (!found) {
		queuedJobs.update((jobs) =>
			jobs.map((j) => {
				if (j.id === jobId) {
					found = true;
					return { ...j, ...updates };
				}
				return j;
			})
		);
	}

	if (!found) {
		jobHistory.update((jobs) =>
			jobs.map((j) => {
				if (j.id === jobId) {
					found = true;
					return { ...j, ...updates };
				}
				return j;
			})
		);
	}

	return found;
}
