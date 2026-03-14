<script lang="ts">
	let {
		current = 0,
		total = 0,
		showLabel = true,
		startedAt = null as number | null,
	}: {
		current?: number;
		total?: number;
		showLabel?: boolean;
		startedAt?: number | null;
	} = $props();

	let pct = $derived(total > 0 ? Math.min(100, (current / total) * 100) : 0);

	let eta = $derived.by(() => {
		if (!startedAt || current <= 0 || total <= 0) return '';
		const elapsed = (Date.now() - startedAt) / 1000;
		const perFrame = elapsed / current;
		const remaining = perFrame * (total - current);
		if (remaining < 1) return '';
		if (remaining < 60) return `~${Math.ceil(remaining)}s`;
		if (remaining < 3600) return `~${Math.ceil(remaining / 60)}m`;
		const h = Math.floor(remaining / 3600);
		const m = Math.ceil((remaining % 3600) / 60);
		return `~${h}h${m}m`;
	});

	let fps = $derived.by(() => {
		if (!startedAt || current <= 0) return '';
		const elapsed = (Date.now() - startedAt) / 1000;
		if (elapsed < 1) return '';
		return (current / elapsed).toFixed(1);
	});
</script>

<div class="progress-wrap">
	<div class="progress-track">
		<div
			class="progress-fill"
			class:indeterminate={total === 0}
			style="width: {pct}%"
		></div>
	</div>
	{#if showLabel && total > 0}
		<div class="progress-stats mono">
			<span class="progress-main">{current}<span class="dim">/{total}</span> &middot; {pct.toFixed(0)}%</span>
			{#if fps}
				<span class="progress-detail">{fps} fps</span>
			{/if}
			{#if eta}
				<span class="progress-detail">{eta} left</span>
			{/if}
		</div>
	{/if}
</div>

<style>
	.progress-wrap {
		display: flex;
		align-items: center;
		gap: var(--sp-3);
		width: 100%;
	}

	.progress-track {
		flex: 1;
		height: 6px;
		background: var(--surface-3);
		border-radius: 3px;
		overflow: hidden;
		box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3);
	}

	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--accent-dim), var(--accent));
		border-radius: 3px;
		transition: width 0.3s ease-out;
		position: relative;
		box-shadow: 0 0 8px rgba(255, 242, 3, 0.25);
	}

	.progress-fill::after {
		content: '';
		position: absolute;
		inset: 0;
		background: linear-gradient(90deg, transparent 60%, rgba(255, 255, 255, 0.2));
		animation: shimmer 2s ease-in-out infinite;
	}

	.progress-fill.indeterminate {
		width: 30% !important;
		animation: indeterminate 1.5s ease-in-out infinite;
	}

	.progress-stats {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 0;
		min-width: 90px;
	}

	.progress-main {
		font-size: 11px;
		color: var(--text-primary);
		white-space: nowrap;
	}

	.dim {
		color: var(--text-tertiary);
	}

	.progress-detail {
		font-size: 10px;
		color: var(--text-secondary);
		white-space: nowrap;
	}

	@keyframes shimmer {
		0%, 100% { opacity: 0; }
		50% { opacity: 1; }
	}

	@keyframes indeterminate {
		0% { transform: translateX(-100%); }
		100% { transform: translateX(400%); }
	}
</style>
