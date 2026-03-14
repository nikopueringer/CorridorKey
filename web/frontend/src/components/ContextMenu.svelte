<script lang="ts">
	import { onMount } from 'svelte';

	export interface MenuItem {
		label: string;
		icon?: string;
		danger?: boolean;
		disabled?: boolean;
		action: () => void;
	}

	let {
		items = [],
		x = 0,
		y = 0,
		visible = $bindable(false),
	}: {
		items: MenuItem[];
		x: number;
		y: number;
		visible: boolean;
	} = $props();

	let menuEl: HTMLDivElement | undefined = $state();

	function handleClick(item: MenuItem) {
		if (item.disabled) return;
		visible = false;
		item.action();
	}

	function onClickOutside(e: MouseEvent) {
		if (menuEl && !menuEl.contains(e.target as Node)) {
			visible = false;
		}
	}

	function onKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') visible = false;
	}

	$effect(() => {
		if (visible) {
			// Adjust position to stay in viewport
			requestAnimationFrame(() => {
				if (!menuEl) return;
				const rect = menuEl.getBoundingClientRect();
				if (rect.right > window.innerWidth) {
					menuEl.style.left = `${x - rect.width}px`;
				}
				if (rect.bottom > window.innerHeight) {
					menuEl.style.top = `${y - rect.height}px`;
				}
			});
		}
	});
</script>

<svelte:window onclick={onClickOutside} onkeydown={onKeydown} />

{#if visible && items.length > 0}
	<div class="context-menu" bind:this={menuEl} style="left: {x}px; top: {y}px">
		{#each items as item}
			{#if item.label === '---'}
				<div class="separator"></div>
			{:else}
				<button
					class="menu-item"
					class:danger={item.danger}
					class:disabled={item.disabled}
					onclick={() => handleClick(item)}
				>
					{item.label}
				</button>
			{/if}
		{/each}
	</div>
{/if}

<style>
	.context-menu {
		position: fixed;
		z-index: 1000;
		min-width: 180px;
		padding: 4px;
		background: var(--surface-3);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.04);
		backdrop-filter: blur(12px);
	}

	.menu-item {
		display: block;
		width: 100%;
		padding: 8px 12px;
		font-size: 13px;
		font-family: inherit;
		text-align: left;
		color: var(--text-primary);
		background: none;
		border: none;
		border-radius: var(--radius-sm);
		cursor: pointer;
		transition: background 0.1s;
	}

	.menu-item:hover {
		background: var(--surface-4);
	}

	.menu-item.danger {
		color: var(--state-error);
	}

	.menu-item.danger:hover {
		background: rgba(255, 82, 82, 0.1);
	}

	.menu-item.disabled {
		color: var(--text-tertiary);
		cursor: not-allowed;
	}

	.menu-item.disabled:hover {
		background: none;
	}

	.separator {
		height: 1px;
		margin: 4px 8px;
		background: var(--border);
	}
</style>
