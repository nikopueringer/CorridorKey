<script lang="ts">
	import { toasts, removeToast } from '$lib/stores/toasts';
</script>

{#if $toasts.length > 0}
	<div class="toast-container">
		{#each $toasts as t (t.id)}
			<div class="toast toast-{t.type}" role="alert">
				<span class="toast-msg">{t.message}</span>
				<button class="toast-close" onclick={() => removeToast(t.id)}>&times;</button>
			</div>
		{/each}
	</div>
{/if}

<style>
	.toast-container {
		position: fixed;
		bottom: 20px;
		right: 20px;
		z-index: 9000;
		display: flex;
		flex-direction: column-reverse;
		gap: 8px;
		max-width: 400px;
	}

	.toast {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 10px 14px;
		border-radius: var(--radius-md, 8px);
		font-size: 13px;
		color: var(--text-primary, #fff);
		background: var(--surface-3, #1c1b17);
		border: 1px solid var(--border, rgba(255, 255, 255, 0.07));
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
		animation: slide-in 0.25s ease-out;
	}

	.toast-success { border-left: 3px solid var(--state-complete, #5dd879); }
	.toast-error { border-left: 3px solid var(--state-error, #ff5252); }
	.toast-warning { border-left: 3px solid var(--state-raw, #f0a030); }
	.toast-info { border-left: 3px solid var(--accent, #fff203); }

	.toast-msg { flex: 1; }

	.toast-close {
		background: none;
		border: none;
		color: var(--text-tertiary, #605f56);
		font-size: 18px;
		cursor: pointer;
		line-height: 1;
		padding: 0 2px;
	}
	.toast-close:hover { color: var(--text-primary, #fff); }

	@keyframes slide-in {
		from { transform: translateX(100%); opacity: 0; }
		to { transform: translateX(0); opacity: 1; }
	}
</style>
