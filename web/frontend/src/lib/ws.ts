/** WebSocket client with auto-reconnect. */

export interface WSMessage {
	type: string;
	data: Record<string, unknown>;
}

type MessageHandler = (msg: WSMessage) => void;

let socket: WebSocket | null = null;
let handlers: MessageHandler[] = [];
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let intentionallyClosed = false;

function getWsUrl(): string {
	const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
	return `${proto}//${window.location.host}/ws`;
}

function doConnect() {
	if (socket?.readyState === WebSocket.OPEN || socket?.readyState === WebSocket.CONNECTING) {
		return;
	}

	socket = new WebSocket(getWsUrl());

	socket.onopen = () => {
		if (reconnectTimer) {
			clearTimeout(reconnectTimer);
			reconnectTimer = null;
		}
	};

	socket.onmessage = (event) => {
		try {
			const msg: WSMessage = JSON.parse(event.data);
			for (const handler of handlers) {
				handler(msg);
			}
		} catch {
			// ignore malformed messages
		}
	};

	socket.onclose = () => {
		socket = null;
		if (!intentionallyClosed) {
			reconnectTimer = setTimeout(doConnect, 2000);
		}
	};

	socket.onerror = () => {
		socket?.close();
	};
}

export function connect() {
	intentionallyClosed = false;
	doConnect();
}

export function disconnect() {
	intentionallyClosed = true;
	if (reconnectTimer) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
	socket?.close();
	socket = null;
}

export function onMessage(handler: MessageHandler): () => void {
	handlers.push(handler);
	return () => {
		handlers = handlers.filter((h) => h !== handler);
	};
}

export function isConnected(): boolean {
	return socket?.readyState === WebSocket.OPEN;
}
