<script lang="ts">
	import type { Clip } from '$lib/api';
	import { api } from '$lib/api';

	let { clip }: { clip: Clip } = $props();

	const stateColors: Record<string, string> = {
		RAW: 'var(--state-raw)',
		READY: 'var(--state-ready)',
		COMPLETE: 'var(--state-complete)',
		ERROR: 'var(--state-error)',
		EXTRACTING: 'var(--state-extracting)',
		MASKED: 'var(--state-masked)',
	};

	let color = $derived(stateColors[clip.state] ?? 'var(--text-tertiary)');
	let thumbUrl = $derived(
		clip.has_outputs
			? api.preview.url(clip.name, 'comp', 0)
			: clip.frame_count > 0
				? api.preview.url(clip.name, 'input', 0)
				: null
	);
</script>

<a href="/clips/{encodeURIComponent(clip.name)}" class="card">
	<div class="card-thumb">
		{#if thumbUrl}
			<img src={thumbUrl} alt="{clip.name} preview" loading="lazy" />
		{:else}
			<div class="thumb-empty">
				<svg width="32" height="32" viewBox="0 0 32 32" fill="none" opacity="0.3">
					<rect x="4" y="7" width="24" height="18" rx="2.5" stroke="currentColor" stroke-width="1.5"/>
					<circle cx="11" cy="14" r="2" stroke="currentColor" stroke-width="1.2"/>
					<path d="M4 20l6-4 4 3 5-5 9 6" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/>
				</svg>
			</div>
		{/if}
		<span class="state-badge mono" style="--badge-color: {color}">{clip.state}</span>
		<div class="card-shine"></div>
	</div>

	<div class="card-body">
		<span class="card-name">{clip.name}</span>
		<div class="card-meta mono">
			<span>{clip.frame_count} frames</span>
			{#if clip.completed_frames > 0}
				<span class="sep">&middot;</span>
				<span style="color: var(--state-complete)">{clip.completed_frames} done</span>
			{/if}
		</div>
		{#if clip.error_message}
			<span class="card-error mono">{clip.error_message}</span>
		{/if}
	</div>
</a>

<style>
	.card {
		display: flex;
		flex-direction: column;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		overflow: hidden;
		transition: all 0.2s ease;
		cursor: pointer;
	}

	.card:hover {
		border-color: var(--border-active);
		transform: translateY(-2px);
		box-shadow:
			0 8px 24px rgba(0, 0, 0, 0.4),
			0 0 0 1px var(--border-active),
			0 0 20px rgba(255, 242, 3, 0.04);
	}

	.card:hover .card-shine {
		opacity: 1;
	}

	.card-thumb {
		position: relative;
		aspect-ratio: 16 / 9;
		background: var(--surface-1);
		overflow: hidden;
	}

	.card-thumb img {
		width: 100%;
		height: 100%;
		object-fit: cover;
		transition: transform 0.3s ease;
	}

	.card:hover .card-thumb img {
		transform: scale(1.02);
	}

	.card-shine {
		position: absolute;
		inset: 0;
		background: linear-gradient(135deg, rgba(255, 242, 3, 0.04) 0%, transparent 50%);
		opacity: 0;
		transition: opacity 0.3s;
		pointer-events: none;
	}

	.thumb-empty {
		width: 100%;
		height: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--text-tertiary);
		background:
			linear-gradient(135deg, var(--surface-1) 0%, var(--surface-2) 100%),
			repeating-linear-gradient(
				45deg,
				transparent,
				transparent 10px,
				rgba(255, 255, 255, 0.01) 10px,
				rgba(255, 255, 255, 0.01) 20px
			);
	}

	.state-badge {
		position: absolute;
		top: var(--sp-2);
		right: var(--sp-2);
		padding: 3px 8px;
		font-size: 9px;
		font-weight: 600;
		letter-spacing: 0.08em;
		color: var(--badge-color);
		background: rgba(0, 0, 0, 0.75);
		border: 1px solid var(--badge-color);
		border-radius: var(--radius-sm);
		backdrop-filter: blur(8px);
		text-shadow: 0 0 8px var(--badge-color);
	}

	.card-body {
		padding: var(--sp-3) var(--sp-4);
		display: flex;
		flex-direction: column;
		gap: 3px;
	}

	.card-name {
		font-weight: 600;
		font-size: 14px;
		color: var(--text-primary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.card-meta {
		font-size: 10px;
		color: var(--text-secondary);
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.sep {
		color: var(--text-tertiary);
	}

	.card-error {
		font-size: 10px;
		color: var(--state-error);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		margin-top: 2px;
	}
</style>
