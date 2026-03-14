<script lang="ts">
	import { currentJob, queuedJobs, jobHistory, refreshJobs } from '$lib/stores/jobs';
	import { api } from '$lib/api';
	import JobRow from '../../components/JobRow.svelte';

	let cancelling = $state(false);

	async function cancelAll() {
		cancelling = true;
		try {
			await api.jobs.cancelAll();
			await refreshJobs();
		} finally {
			cancelling = false;
		}
	}

	let hasActive = $derived($currentJob !== null || $queuedJobs.length > 0);
</script>

<svelte:head>
	<title>Jobs — CorridorKey</title>
</svelte:head>

<div class="page">
	<div class="page-header">
		<h1 class="page-title">Jobs</h1>
		<div class="header-actions">
			<button class="btn-ghost" onclick={() => refreshJobs()}>
				<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
					<path d="M12 7a5 5 0 11-1.5-3.5M12 2v3h-3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
				</svg>
				Refresh
			</button>
			{#if hasActive}
				<button class="btn-ghost btn-danger" onclick={cancelAll} disabled={cancelling}>
					Cancel All
				</button>
			{/if}
		</div>
	</div>

	<!-- Current Job -->
	{#if $currentJob}
		<section class="section">
			<h2 class="section-title mono">RUNNING</h2>
			<div class="job-list">
				<JobRow job={$currentJob} showCancel />
			</div>
		</section>
	{/if}

	<!-- Queued -->
	{#if $queuedJobs.length > 0}
		<section class="section">
			<h2 class="section-title mono">QUEUED <span class="count">{$queuedJobs.length}</span></h2>
			<div class="job-list">
				{#each $queuedJobs as job (job.id)}
					<JobRow {job} showCancel />
				{/each}
			</div>
		</section>
	{/if}

	<!-- History -->
	{#if $jobHistory.length > 0}
		<section class="section">
			<h2 class="section-title mono">HISTORY</h2>
			<div class="job-list">
				{#each $jobHistory as job (job.id)}
					<JobRow {job} />
				{/each}
			</div>
		</section>
	{/if}

	{#if !$currentJob && $queuedJobs.length === 0 && $jobHistory.length === 0}
		<div class="empty-state">
			<svg width="48" height="48" viewBox="0 0 48 48" fill="none">
				<path d="M8 16l16 9.5L40 16M8 24l16 9.5L40 24M8 32l16 9.5L40 32M8 8L24 17.5 40 8 24 0 8 8z" stroke="var(--text-tertiary)" stroke-width="1.5" stroke-linejoin="round"/>
			</svg>
			<p class="empty-text">No jobs</p>
			<p class="empty-hint mono">Submit a job from a clip's detail page.</p>
		</div>
	{/if}
</div>

<style>
	.page {
		padding: var(--sp-5) var(--sp-6);
		display: flex;
		flex-direction: column;
		gap: var(--sp-4);
	}

	.page-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.page-title {
		font-family: var(--font-sans);
		font-size: 20px;
		font-weight: 700;
		letter-spacing: -0.01em;
	}

	.header-actions {
		display: flex;
		gap: var(--sp-2);
	}

	.btn-ghost {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
		padding: var(--sp-2) var(--sp-3);
		font-size: 12px;
		font-weight: 500;
		color: var(--text-secondary);
		background: transparent;
		border: 1px solid var(--border);
		border-radius: 6px;
		cursor: pointer;
		transition: all 0.15s;
	}

	.btn-ghost:hover {
		color: var(--text-primary);
		border-color: var(--text-tertiary);
		background: var(--surface-2);
	}

	.btn-ghost:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-danger {
		color: var(--state-error);
		border-color: rgba(255, 82, 82, 0.3);
	}

	.btn-danger:hover {
		color: var(--state-error) !important;
		background: rgba(255, 82, 82, 0.08) !important;
		border-color: rgba(255, 82, 82, 0.5) !important;
	}

	.section {
		display: flex;
		flex-direction: column;
		gap: 0;
	}

	.section-title {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
		padding: var(--sp-2) var(--sp-4);
		background: var(--surface-1);
		border: 1px solid var(--border);
		border-radius: 8px 8px 0 0;
		display: flex;
		align-items: center;
		gap: var(--sp-2);
	}

	.count {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: 16px;
		height: 16px;
		padding: 0 4px;
		font-size: 9px;
		background: var(--surface-4);
		border-radius: 8px;
		color: var(--text-secondary);
	}

	.job-list {
		border: 1px solid var(--border);
		border-top: none;
		border-radius: 0 0 8px 8px;
		overflow: hidden;
		background: var(--surface-1);
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--sp-3);
		padding: var(--sp-8) 0;
		text-align: center;
	}

	.empty-text {
		font-size: 15px;
		font-weight: 500;
		color: var(--text-secondary);
	}

	.empty-hint {
		font-size: 11px;
		color: var(--text-tertiary);
		max-width: 300px;
	}
</style>
