<script lang="ts">
	import type { Job } from '$lib/api';
	import { api } from '$lib/api';
	import { jobStartedAt } from '$lib/stores/jobs';
	import ProgressBar from './ProgressBar.svelte';

	let { job, showCancel = false }: { job: Job; showCancel?: boolean } = $props();

	const typeLabels: Record<string, string> = {
		inference: 'Inference',
		gvm_alpha: 'GVM Alpha',
		videomama_alpha: 'VideoMaMa',
		preview_reprocess: 'Preview',
		video_extract: 'Extract',
		video_stitch: 'Stitch',
	};

	const statusColors: Record<string, string> = {
		running: 'var(--state-running)',
		queued: 'var(--state-queued)',
		completed: 'var(--state-complete)',
		cancelled: 'var(--state-cancelled)',
		failed: 'var(--state-failed)',
	};

	let label = $derived(typeLabels[job.job_type] ?? job.job_type);
	let statusColor = $derived(statusColors[job.status] ?? 'var(--text-tertiary)');
	let isRunning = $derived(job.status === 'running');
	let isFailed = $derived(job.status === 'failed');
	let expanded = $state(false);
	let logDetail = $state<string | null>(null);

	async function handleCancel() {
		await api.jobs.cancel(job.id);
	}

	async function toggleLog() {
		if (!isFailed) return;
		expanded = !expanded;
		if (expanded && !logDetail) {
			try {
				const log = await api.jobs.getLog(job.id);
				logDetail = JSON.stringify(log, null, 2);
			} catch {
				logDetail = job.error_message ?? 'No details available';
			}
		}
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="job-row" class:running={isRunning} class:failed={isFailed} onclick={toggleLog}>
	<div class="job-type mono">
		<span class="type-dot" style="background: {statusColor}; box-shadow: 0 0 6px {statusColor}"></span>
		{label}
	</div>

	<div class="job-info">
		<span class="job-clip">{job.clip_name}</span>
		{#if isRunning}
			<ProgressBar current={job.current_frame} total={job.total_frames} startedAt={$jobStartedAt} />
		{:else}
			<span class="job-status mono" style="color: {statusColor}">{job.status.toUpperCase()}</span>
		{/if}
	</div>

	<div class="job-actions">
		<span class="job-id mono">{job.id}</span>
		{#if showCancel && (job.status === 'running' || job.status === 'queued')}
			<button class="cancel-btn" onclick={handleCancel} title="Cancel job">
				<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
					<path d="M3.5 3.5l7 7M10.5 3.5l-7 7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
				</svg>
			</button>
		{/if}
	</div>

	{#if job.error_message}
		<div class="job-error mono">
			{job.error_message}
			{#if isFailed}
				<span class="expand-hint">{expanded ? '▲ collapse' : '▼ click for details'}</span>
			{/if}
		</div>
	{/if}
	{#if expanded && logDetail}
		<pre class="job-log mono">{logDetail}</pre>
	{/if}
</div>

<style>
	.job-row {
		display: grid;
		grid-template-columns: 110px 1fr auto;
		gap: var(--sp-3);
		align-items: center;
		padding: var(--sp-3) var(--sp-4);
		border-bottom: 1px solid var(--border-subtle);
		transition: background 0.15s;
	}

	.job-row:last-child {
		border-bottom: none;
	}

	.job-row:hover {
		background: var(--surface-2);
	}

	.job-row.running {
		background: linear-gradient(90deg, rgba(255, 242, 3, 0.06), transparent);
		border-left: 3px solid var(--accent);
		box-shadow: inset 0 0 20px rgba(255, 242, 3, 0.02);
	}

	.job-type {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
		font-size: 11px;
		font-weight: 500;
		color: var(--text-secondary);
	}

	.type-dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.job-info {
		display: flex;
		flex-direction: column;
		gap: 4px;
		min-width: 0;
	}

	.job-clip {
		font-size: 13px;
		font-weight: 600;
		color: var(--text-primary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.job-status {
		font-size: 10px;
		letter-spacing: 0.06em;
		font-weight: 500;
	}

	.job-actions {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
	}

	.job-id {
		font-size: 9px;
		color: var(--text-tertiary);
	}

	.cancel-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 26px;
		height: 26px;
		border-radius: var(--radius-sm);
		border: 1px solid var(--border);
		background: var(--surface-3);
		color: var(--text-secondary);
		cursor: pointer;
		transition: all 0.15s;
	}

	.cancel-btn:hover {
		color: var(--state-error);
		border-color: var(--state-error);
		background: rgba(255, 82, 82, 0.1);
		box-shadow: 0 0 8px rgba(255, 82, 82, 0.1);
	}

	.job-row.failed {
		cursor: pointer;
	}

	.job-error {
		grid-column: 1 / -1;
		font-size: 11px;
		color: var(--state-error);
		padding: var(--sp-1) 0 0 calc(110px + var(--sp-3));
		display: flex;
		align-items: baseline;
		gap: var(--sp-2);
	}

	.expand-hint {
		font-size: 9px;
		color: var(--text-tertiary);
		flex-shrink: 0;
	}

	.job-log {
		grid-column: 1 / -1;
		font-size: 10px;
		color: var(--text-secondary);
		background: var(--surface-1);
		padding: var(--sp-3);
		border-radius: var(--radius-sm);
		max-height: 200px;
		overflow: auto;
		white-space: pre-wrap;
		word-break: break-all;
		margin: var(--sp-2) 0 0 calc(110px + var(--sp-3));
	}
</style>
