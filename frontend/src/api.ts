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

export interface EvidenceLinks {
  available: boolean;
  confirmation_ready: boolean;
  integrity_status: string;
  integrity_anchor: string;
  package_sha256: string | null;
  integrity_errors: string[];
  confirmation_blockers: string[];
  original_url: string | null;
  annotated_url: string | null;
  clip_url: string | null;
  trace_url: string | null;
  hashes: Record<string, string>;
}

export interface ReviewRecord {
  previous_status: string;
  new_status: string;
  notes: string | null;
  reviewer_id: string;
  created_at: string;
}

export interface EventDetail extends SafetyEvent {
  vehicle_type: string;
  fusion_score: number | null;
  temporal_score: number | null;
  evidence: EvidenceLinks;
  evidence_trace: Record<string, unknown>;
  review_history: ReviewRecord[];
}

export interface V32Summary {
  status: string;
  ai85_gate_met: boolean;
  track_a_f1: number;
  track_a_precision: number;
  track_a_recall: number;
  track_b_f1: number;
  track_c_f1: number;
  seatbelt_binary_accuracy: number;
  seatbelt_binary_macro_f1: number;
  seatbelt_3class_balanced_accuracy: number;
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
    activation_state?: string;
  };
  runtime: {
    fusion: {
      fusion_enabled: boolean;
      fusion_mode: string;
      fusion_available: boolean;
      fusion_fail_closed: boolean;
      fusion_artifact_sha256: string | null;
      fusion_threshold: number;
    };
  };
  event_metrics_status: string;
  v32_summary?: V32Summary | null;
}

export interface TrackAggregateMetrics {
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  total_gt_events: number;
  precision: number;
  recall: number;
  f1: number;
  false_alarms_per_minute: number;
  evaluated_minutes?: number;
}

export interface TrackProtocolResult {
  aggregate: TrackAggregateMetrics;
  per_subject: Record<string, TrackAggregateMetrics>;
}

export interface MatchedEventRecord {
  prediction: {
    event_type: string;
    start_seconds: number;
    end_seconds: number;
    confidence: number;
    duration: number;
    posture?: string;
    post_call_split?: boolean;
    subject_id: string;
    source_action?: string;
  };
  ground_truth: {
    source_id: string;
    source_group: string;
    source_participant_id: string;
    source_session_id: string;
    source_action: string;
    development_phone_state: string;
    start_frame: number;
    end_frame: number;
    start_seconds: number;
    end_seconds: number;
    duration_seconds: number;
  };
  iou: number;
  start_latency: number;
  end_latency: number;
}

export interface CausalDecisionRecord {
  sequence: number;
  decision_type: "START" | "END";
  reason: string;
  decision_time_seconds: number;
  evidence_available_through_seconds: number;
  event_type: string;
  event_start_seconds: number;
  event_end_seconds: number | null;
  last_confirmed_end_seconds: number;
  confidence: number;
  vehicle_context_id: string;
  track_id: string;
  occupant_group: string;
  subject_id: string;
}

export interface SubjectBenchmarkDetail {
  subject_id: string;
  duration_seconds: number;
  metrics: {
    true_positives: number;
    false_positives: number;
    false_negatives: number;
    total_gt_events: number;
    evaluated_minutes: number;
    precision: number;
    recall: number;
    f1: number;
    false_alarms_per_minute: number;
    matched_events: MatchedEventRecord[];
    unmatched_predictions: MatchedEventRecord["prediction"][];
    unmatched_gt: MatchedEventRecord["ground_truth"][];
    processed_frames: number;
    wall_time_seconds: number;
    effective_fps: number;
  };
  decisions: CausalDecisionRecord[];
}

export interface SeatbeltClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface SeatbeltSplitMetrics {
  confusion_matrix: Record<string, Record<string, number>>;
  per_class: Record<string, SeatbeltClassMetrics>;
  accuracy: number;
  balanced_accuracy: number;
  macro_f1: number;
}

export interface SeatbeltV32Evaluation {
  dataset: string;
  train_reference_count: number;
  val_evaluation_count: number;
  test_split_accessed: boolean;
  dmd_data_used: boolean;
  binary_fastened_vs_unfastened_616_val: {
    sample_count: number;
    accuracy: number;
    macro_f1: number;
    seatbelt_fastened: SeatbeltClassMetrics;
    seatbelt_unfastened: SeatbeltClassMetrics;
  };
  three_class_s2_baseline_641_val: {
    sample_count: number;
    raw: SeatbeltSplitMetrics;
    post_policy: SeatbeltSplitMetrics;
    top1_ece_10_bin: number;
  };
  three_class_s3_v32_calibrated_641_val: {
    sample_count: number;
    raw: SeatbeltSplitMetrics;
    post_policy: SeatbeltSplitMetrics;
    top1_ece_10_bin: number;
    abstention: {
      output_uncertain_count: number;
      output_uncertain_rate: number;
      threshold_forced_count: number;
      threshold_forced_rate: number;
    };
  };
}

export interface KaggleExecutionReceipt {
  generated_at: string;
  dataset: {
    ref: string;
    url: string;
    visibility: string;
    code_manifest_sha256: string;
    contains_dmd_media: boolean;
    contains_credentials: boolean;
  };
  kernel: {
    ref: string;
    url: string;
    version: number;
    status: string;
  };
  verified_result: {
    status: string;
    scope: string;
    gpu: string;
    candidate_identity_test: string;
    v32_normalized_proximity_test: string;
    v32_aspect_ratio_clutter_filter_test: string;
    device_provenance: Record<string, string>;
    stage_runtime: Record<string, { calls: number; p50_ms: number; p95_ms: number; mean_ms: number }>;
    peak_cuda_allocated_bytes: number;
    raw_d5_status: string;
    development_ai85_status: string;
  };
}

export interface BugFixRecord {
  id: string;
  title: string;
  module: string;
  root_cause: string;
  fix: string;
  impact: string;
}

export interface V32BenchmarkPayload {
  available: boolean;
  schema_version: string;
  status: string;
  created_at: string;
  frozen_policy: {
    path: string;
    sha256: string;
    declared_policy_sha256: string;
    parameters: {
      iou_threshold: number;
      onset_tolerance_seconds: number;
      offset_tolerance_seconds: number;
      containment_overlap_threshold: number;
      containment_min_duration_seconds: number;
    };
  };
  v31_baseline: {
    source: string;
    metrics: {
      true_positives: number;
      false_positives: number;
      false_negatives: number;
      precision: number;
      recall: number;
      f1: number;
    };
  };
  v32_results: {
    source_run: string;
    event_matches_sha256: string;
    recomputed_matches_reported: boolean;
    ai85_development_gate_met: boolean;
    track_a_v2_temporal_frozen: TrackProtocolResult;
    track_b_class_aware_exact_4class: TrackProtocolResult;
    track_c_generic_phone_episode_gap3s: TrackProtocolResult;
  };
  live_decision_causality_audit: {
    status: string;
    per_subject: Record<
      string,
      {
        status: string;
        decisions: number;
        starts: number;
        ends: number;
        session_finalizations: number;
        activation_latency_seconds: { mean: number; median: number; maximum: number };
        decision_duration_seconds: { mean: number; median: number; maximum: number };
        errors: string[];
      }
    >;
  };
  governance: {
    dmd_execution_environment: string;
    dmd_media_uploaded_to_kaggle: boolean;
    consumed_holdouts_accessed: string[];
    holdout_note: string;
  };
  subjects: Record<string, SubjectBenchmarkDetail>;
  seatbelt: SeatbeltV32Evaluation;
  kaggle_gpu: KaggleExecutionReceipt;
  five_bug_fixes: BugFixRecord[];
}

export interface AnalysisJobStatus {
  id: string;
  video_id: string;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
  progress: number;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface VideoListItem {
  id: string;
  original_name: string;
  size_bytes: number;
  mime_type: string;
  input_scope: string;
  created_at: string;
  streamable: boolean;
  event_count: number;
  detection_count: number;
  latest_job: {
    id: string;
    status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
    progress: number;
    error: string | null;
    completed_at: string | null;
  } | null;
}

export interface VideoDetectionItem {
  id: string;
  frame_number: number;
  timestamp_seconds: number;
  class_name: string;
  confidence: number;
  bbox: {
    xyxy?: number[];
    phone_context?: string;
    phone_posture?: string;
    source_action?: string;
    pose_confidence?: number;
    [key: string]: unknown;
  };
  track_id: number | null;
  driver_associated: boolean;
  occupant_role: string | null;
  vehicle_context_id: string | null;
}

export interface VideoAnalysisSummary {
  video: {
    id: string;
    original_name: string;
    size_bytes: number;
    mime_type: string;
    input_scope: string;
    created_at: string;
    streamable: boolean;
  };
  job: {
    id: string;
    status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
    progress: number;
    error: string | null;
    created_at: string;
    completed_at: string | null;
  } | null;
  summary: {
    total_detections: number;
    phone_detections: number;
    seatbelt_detections: number;
    occupant_detections: number;
    total_events: number;
    phone_events: number;
    no_seatbelt_events: number;
  };
  events: EventDetail[];
  detections: VideoDetectionItem[];
}

export type PhoneVisibility = "PHONE_VISIBLE" | "NO_PHONE_VISIBLE" | "UNRESOLVED";
export type PhoneUseAssessment = "PHONE_USE_CONFIRMED" | "NOT_PHONE_USE" | "UNRESOLVED";

export interface PhoneProposal {
  proposal_id: string;
  frame_index: number;
  timestamp_seconds: number;
  source: "specialist" | "coco_cell_phone";
  confidence: number;
  geometry_context: string;
  hand_proximity: number;
  face_proximity: number;
  nearest_cross_model_iou: number;
  xyxy_in_driver_roi: number[];
}

export interface PhoneReviewDecision {
  revision: string;
  reviewer_id: string;
  reviewed_at_utc: string;
  phone_visibility: PhoneVisibility;
  phone_use_assessment: PhoneUseAssessment;
  selected_proposal_id: string;
  notes: string;
  review_assistance?: { type: "AI_PRE_REVIEW"; reviewer_id: string; revision: string; accepted_without_changes: boolean };
}

export interface PhonePreReview {
  reviewer_type: "AI";
  reviewer_id: string;
  status: "AI_PRE_REVIEW";
  phone_visibility: PhoneVisibility;
  phone_use_assessment: PhoneUseAssessment;
  selected_proposal_id: string;
  confidence: "HIGH" | "MEDIUM" | "LOW";
  rationale: string;
  revision: string;
}

export interface PhoneReviewCase {
  case_id: string;
  subject_id: string;
  sample_count: number;
  specialist_proposals: number;
  auxiliary_proposals: number;
  cross_model_agreement: boolean;
  auxiliary_review_candidate: boolean;
  pre_review_available: boolean;
  review_status: "PENDING" | "RECHECK" | "REVIEWED";
  decision: PhoneReviewDecision | null;
}

export interface PhoneReviewQueue {
  scope: string;
  cases: PhoneReviewCase[];
  summary: { total: number; reviewed: number; pending: number; recheck: number };
}

export interface PhoneReviewDetail extends Omit<PhoneReviewCase, "review_status"> {
  contact_sheet_sha256: string;
  proposals: PhoneProposal[];
  pre_review: PhonePreReview | null;
  pre_review_changed_since_decision: boolean;
}

export interface PhoneReviewInput {
  phone_visibility: PhoneVisibility;
  phone_use_assessment: PhoneUseAssessment;
  selected_proposal_id: string;
  notes: string;
  expected_revision: string | null;
  pre_review_revision: string | null;
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
  event(id: string) { return request<EventDetail>(`/events/${id}`); },
  statistics() { return request<Statistics>("/statistics"); },
  benchmarkV32() { return request<V32BenchmarkPayload>("/benchmark/v32"); },
  seedV32DemoEvents() { return request<{ seeded_videos: number; seeded_events: number; status: string }>("/benchmark/v32/seed-demo-events", { method: "POST" }); },
  job(id: string) { return request<AnalysisJobStatus>(`/jobs/${id}`); },
  videos() { return request<VideoListItem[]>("/videos"); },
  videoAnalysisSummary(videoId: string) { return request<VideoAnalysisSummary>(`/videos/${videoId}/analysis-summary`); },
  videoStreamUrl(videoId: string, cacheBust?: string | null) {
    const q = cacheBust ? `?v=${encodeURIComponent(cacheBust)}` : "";
    return `${API}/videos/${videoId}/stream${q}`;
  },
  async videoStreamBlob(videoId: string, signal?: AbortSignal) {
    const response = await fetch(`${API}/videos/${videoId}/stream`, { headers: authHeaders(), signal });
    if (!response.ok) throw new Error(`Không thể tải luồng video (${response.status})`);
    return response.blob();
  },
  phoneReviews() { return request<PhoneReviewQueue>("/research/phone-reviews"); },
  phoneReview(caseId: string) { return request<PhoneReviewDetail>(`/research/phone-reviews/${caseId}`); },
  savePhoneReview(caseId: string, payload: PhoneReviewInput) { return request<PhoneReviewDecision>(`/research/phone-reviews/${caseId}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); },
  async phoneReviewImage(caseId: string, signal?: AbortSignal) {
    const response = await fetch(`${API}/research/phone-reviews/${caseId}/contact-sheet`, { headers: authHeaders(), signal });
    if (!response.ok) throw new Error(`Review image could not be loaded (${response.status})`);
    return response.blob();
  },
  review(id: string, status: ReviewStatus, notes?: string) { return request<SafetyEvent>(`/events/${id}/review`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status, notes }) }); },
  async upload(file: File, inputScope: "VEHICLE_CABIN_CROP" | "TRAFFIC_SCENE_WITH_VEHICLE_ROIS" = "VEHICLE_CABIN_CROP") {
    const form = new FormData();
    form.append("upload", file);
    form.append("input_scope", inputScope);
    return request<{ id: string; original_name: string }>("/videos", { method: "POST", body: form });
  },
  analyze(videoId: string) { return request<{ id: string }>(`/videos/${videoId}/analyze`, { method: "POST" }); },
  async evidenceBlob(path: string, signal?: AbortSignal) {
    const response = await fetch(`${API}${path}`, { headers: authHeaders(), signal });
    if (!response.ok) throw new Error(`Evidence could not be loaded (${response.status})`);
    return response.blob();
  },
  async exportEvents() {
    const response = await fetch(`${API}/exports/events.csv`, { headers: authHeaders() });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(body.detail ?? "Export failed");
    }
    const disposition = response.headers.get("Content-Disposition") ?? "";
    const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1] ?? "events.csv";
    return { blob: await response.blob(), filename };
  },
};
