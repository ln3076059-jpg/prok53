export type ReviewStatus = "PENDING" | "CONFIRMED" | "REJECTED" | "NEEDS_REVIEW";
export type EventType = "PHONE" | "NO_SEATBELT";

export interface SafetyEvent {
  id: string;
  video_id: string;
  event_type: EventType;
  confidence: number;
  frame_number: number;
  timestamp_seconds: number;
  track_id: number | null;
  occupant_role: string;
  vehicle_context_id: string;
  model_version: string;
  review_status: ReviewStatus;
  created_at: string;
}

export interface Statistics {
  analyzed_videos: number;
  events_by_type: Record<EventType, number>;
  events_by_review_status: Record<ReviewStatus, number>;
  model: {
    available: boolean;
    loaded: boolean;
    model_version: string;
    weights: string;
    threshold_status?: string;
  };
  event_metrics_status: string;
}

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function authHeaders(): HeadersInit {
  const token = sessionStorage.getItem("roadwatch-token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API}${path}`, { ...options, headers: { ...authHeaders(), ...options.headers } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export const api = {
  async login(email: string, password: string) {
    const token = await request<{ access_token: string }>("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
    sessionStorage.setItem("roadwatch-token", token.access_token);
  },
  events(query = "") { return request<SafetyEvent[]>(`/events${query}`); },
  event(id: string) { return request<SafetyEvent>(`/events/${id}`); },
  statistics() { return request<Statistics>("/statistics"); },
  review(id: string, status: ReviewStatus, notes?: string) { return request<SafetyEvent>(`/events/${id}/review`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status, notes }) }); },
  async upload(file: File) {
    const form = new FormData();
    form.append("upload", file);
    form.append("input_scope", "VEHICLE_CABIN_CROP");
    return request<{ id: string; original_name: string }>("/videos", { method: "POST", body: form });
  },
  analyze(videoId: string) { return request<{ id: string }>(`/videos/${videoId}/analyze`, { method: "POST" }); },
  exportUrl: `${API}/exports/events.csv`,
};
