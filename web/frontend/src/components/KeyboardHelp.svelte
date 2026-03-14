<script lang="ts">
	let visible = $state(false);

	function onKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
		if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
			e.preventDefault();
			visible = !visible;
		}
		if (e.key === 'Escape' && visible) {
			visible = false;
		}
	}

	const shortcuts = [
		{ keys: '?', desc: 'Toggle this help' },
		{ keys: 'Space', desc: 'Play/pause (frame viewer)' },
		{ keys: '← →', desc: 'Previous/next frame' },
		{ keys: 'Home / End', desc: 'First/last frame' },
		{ keys: 'Esc', desc: 'Close dialogs' },
	];
</script>

<svelte:window onkeydown={onKeydown} />

{#if visible}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="kb-overlay" onclick={() => { visible = false; }}>
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="kb-panel" onclick={(e) => e.stopPropagation()}>
			<div class="kb-header">
				<h2>Keyboard Shortcuts</h2>
				<button class="kb-close" onclick={() => { visible = false; }}>&times;</button>
			</div>
			<div class="kb-list">
				{#each shortcuts as s}
					<div class="kb-row">
						<kbd class="mono">{s.keys}</kbd>
						<span>{s.desc}</span>
					</div>
				{/each}
			</div>
		</div>
	</div>
{/if}

<style>
	.kb-overlay {
		position: fixed;
		inset: 0;
		z-index: 8000;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(0, 0, 0, 0.7);
		backdrop-filter: blur(4px);
	}

	.kb-panel {
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: var(--sp-5);
		min-width: 320px;
		box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
	}

	.kb-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: var(--sp-4);
	}

	.kb-header h2 {
		font-size: 16px;
		font-weight: 700;
	}

	.kb-close {
		background: none;
		border: none;
		color: var(--text-tertiary);
		font-size: 22px;
		cursor: pointer;
		line-height: 1;
	}
	.kb-close:hover { color: var(--text-primary); }

	.kb-list {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.kb-row {
		display: flex;
		align-items: center;
		gap: var(--sp-4);
		font-size: 14px;
	}

	kbd {
		display: inline-block;
		min-width: 70px;
		padding: 3px 8px;
		font-size: 12px;
		text-align: center;
		background: var(--surface-4);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--accent);
	}
</style>
