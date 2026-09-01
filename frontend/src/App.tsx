import { FormEvent, useEffect, useRef, useState } from "react";
import {
  Button,
  FileUploaderDropContainer,
  InlineLoading,
  InlineNotification,
  Select,
  SelectItem,
  Tag,
  TextArea,
  TextInput,
} from "@carbon/react";
import {
  ChartBar,
  CheckmarkOutline,
  Dashboard,
  DocumentExport,
  Events,
  Information,
  Login as LoginIcon,
  MachineLearningModel,
  Menu,
  Phone,
  Play,
  Settings as SettingsIcon,
  Upload,
  Video,
  WarningAlt,
} from "@carbon/icons-react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import { api, type EventDetail, type ReviewStatus, type SafetyEvent, type Statistics } from "./api";

const emptyStats: Statistics = {
  analyzed_videos: 0,
  events_by_type: { PHONE: 0, NO_SEATBELT: 0 },
  events_by_review_status: { PENDING: 0, CONFIRMED: 0, REJECTED: 0, NEEDS_REVIEW: 0 },
  model: { available: false, loaded: false, model_version: "UNTRAINED", weights: "models/active/best.pt" },
  runtime: { fusion: { fusion_enabled: false, fusion_mode: "DISABLED", fusion_available: false, fusion_fail_closed: false, fusion_artifact_sha256: null, fusion_threshold: 0.5 } },
  event_metrics_status: "NOT_RUN",
};

function formatTimestamp(seconds: number) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remaining = Math.floor(seconds % 60);
  return [hours, minutes, remaining].map((part) => String(part).padStart(2, "0")).join(":");
}

function StatusTag({ status }: { status: ReviewStatus }) {
  const types: Record<ReviewStatus, "gray" | "green" | "red" | "warm-gray"> = {
    PENDING: "gray",
    CONFIRMED: "green",
    REJECTED: "red",
    NEEDS_REVIEW: "warm-gray",
  };
  return <Tag type={types[status]}>{status.replace("_", " ")}</Tag>;
}

function useEvidenceAssets(event: EventDetail | null) {
  const [assets, setAssets] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    const objectUrls: string[] = [];
    let active = true;
    if (!event || ![event.evidence.original_url, event.evidence.annotated_url, event.evidence.clip_url].some(Boolean)) {
      setAssets({});
      setLoading(false);
      setError("");
      return () => controller.abort();
    }
    const entries = [
      ["original", event.evidence.original_url],
      ["annotated", event.evidence.annotated_url],
      ["clip", event.evidence.clip_url],
    ].filter((entry): entry is [string, string] => Boolean(entry[1]));
    setLoading(true);
    setError("");
    Promise.all(entries.map(async ([key, path]) => {
      const blob = await api.evidenceBlob(path, controller.signal);
      const url = URL.createObjectURL(blob);
      objectUrls.push(url);
      return [key, url] as const;
    }))
      .then((rows) => { if (active) setAssets(Object.fromEntries(rows)); })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        if (active) setError(reason instanceof Error ? reason.message : "Evidence could not be loaded");
      })
      .finally(() => { if (active) setLoading(false); });
    return () => {
      active = false;
      controller.abort();
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [event]);
  return { assets, loading, error };
}

function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      await api.login(email, password);
      navigate("/");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="login-page">
      <section className="login-record" aria-labelledby="login-title">
        <div className="brand-mark" aria-hidden="true"><CheckmarkOutline size={30} /></div>
        <p className="product-name">Roadwatch</p>
        <h1 id="login-title">Review driver safety evidence</h1>
        <p className="login-copy">Sign in to analyze recordings, inspect evidence, and record review decisions.</p>
        {error && <InlineNotification kind="error" title="Sign-in failed" subtitle={error} lowContrast hideCloseButton />}
        <form onSubmit={submit} className="login-form">
          <TextInput id="email" labelText="Email" type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required />
          <TextInput id="password" labelText="Password" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required />
          <Button type="submit" renderIcon={LoginIcon} disabled={loading}>{loading ? "Signing in" : "Sign in"}</Button>
        </form>
        <p className="setup-note">First installation? Create an administrator with <code>python tools/create_admin.py</code>.</p>
      </section>
      <aside className="login-principles" aria-label="System principles">
        <h2>Evidence before inference</h2>
        <dl>
          <div><dt>Detector</dt><dd>One three-class YOLO11s model</dd></div>
          <div><dt>Events</dt><dd>Driver-associated and confirmed over time</dd></div>
          <div><dt>Uncertainty</dt><dd>Preserved for human review</dd></div>
        </dl>
      </aside>
    </main>
  );
}

const navigation = [
  ["/", "Overview", Dashboard],
  ["/upload", "Upload / Analysis", Upload],
  ["/events", "Events", Events],
  ["/statistics", "Statistics", ChartBar],
  ["/models", "Models / About", MachineLearningModel],
  ["/settings", "Settings", SettingsIcon],
] as const;

function Shell({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const [mobile, setMobile] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const firstLink = useRef<HTMLAnchorElement>(null);
  useEffect(() => {
    const query = window.matchMedia("(max-width: 760px)");
    const update = () => setMobile(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    if (!mobile || !open) return;
    firstLink.current?.focus();
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setOpen(false); menuButton.current?.focus(); }
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [mobile, open]);
  const closeDrawer = () => { setOpen(false); if (mobile) menuButton.current?.focus(); };
  return (
    <div className="app-shell">
      {mobile && open && <button className="drawer-backdrop" aria-label="Close navigation" onClick={closeDrawer} />}
      <aside className={`side-rail ${open ? "is-open" : ""}`} inert={mobile && !open} aria-hidden={mobile && !open}>
        <Link className="rail-brand" to="/" onClick={() => setOpen(false)}>
          <span className="rail-symbol"><CheckmarkOutline size={22} /></span>
          <span>Roadwatch</span>
        </Link>
        <nav aria-label="Primary">
          {navigation.map(([href, label, Icon], index) => (
            <Link ref={index === 0 ? firstLink : undefined} key={href} to={href} className={location.pathname === href ? "active" : ""} onClick={() => setOpen(false)}>
              <Icon size={20} /><span>{label}</span>
            </Link>
          ))}
        </nav>
        <div className="rail-foot"><span>Scientific status</span><strong>Pre-training</strong></div>
        <button className="drawer-close" type="button" onClick={closeDrawer}>Close navigation</button>
      </aside>
      <header className="mobile-header">
        <button ref={menuButton} type="button" aria-label={open ? "Close navigation" : "Open navigation"} aria-expanded={open} onClick={() => setOpen((value) => !value)}><Menu size={22} /></button>
        <span>Roadwatch</span>
      </header>
      <div className="page-shell" inert={mobile && open}>{children}</div>
    </div>
  );
}

function PageHeader({ title, description, action }: { title: string; description: string; action?: React.ReactNode }) {
  return <header className="page-header"><div><h1>{title}</h1><p>{description}</p></div>{action}</header>;
}

function useDashboardData() {
  const [stats, setStats] = useState<Statistics | null>(null);
  const [events, setEvents] = useState<SafetyEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([api.statistics(), api.events("?limit=8")])
      .then(([statistics, recent]) => { setStats(statistics); setEvents(recent); })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Data could not be loaded"))
      .finally(() => setLoading(false));
  }, []);
  return { stats, events, loading, error };
}

function MetricStrip({ stats }: { stats: Statistics }) {
  const metrics = [
    ["Analyzed videos", stats.analyzed_videos, Video],
    ["Phone events", stats.events_by_type.PHONE, Phone],
    ["No seatbelt", stats.events_by_type.NO_SEATBELT, WarningAlt],
    ["Needs review", stats.events_by_review_status.NEEDS_REVIEW + stats.events_by_review_status.PENDING, Information],
  ] as const;
  return <section className="metric-strip" aria-label="Operational summary">{metrics.map(([label, value, Icon]) => <div key={label}><Icon size={22} /><span>{label}</span><strong>{value}</strong></div>)}</section>;
}

function EventTable({ events, compact = false }: { events: SafetyEvent[]; compact?: boolean }) {
  if (!events.length) return <EmptyState title="No events recorded" copy="Upload and analyze a video after installing a locked model." />;
  return (
    <div className="table-scroll"><table className="event-table"><thead><tr><th>Type</th><th>Time</th><th>Confidence</th>{!compact && <th>Video</th>}<th>Review</th><th><span className="visually-hidden">Open</span></th></tr></thead>
      <tbody>{events.map((event) => <tr key={event.id}><td><span className={`event-kind ${event.event_type.toLowerCase()}`}>{event.event_type.replace("_", " ")}</span></td><td className="number">{formatTimestamp(event.timestamp_seconds)}</td><td className="number">{Math.round(event.confidence * 100)}%</td>{!compact && <td className="mono">{event.video_id.slice(0, 8)}</td>}<td><StatusTag status={event.review_status} /></td><td><Link className="row-link" to={`/events/${event.id}`}>Inspect</Link></td></tr>)}</tbody>
    </table></div>
  );
}

function EmptyState({ title, copy }: { title: string; copy: string }) {
  return <div className="empty-state"><Events size={32} /><h3>{title}</h3><p>{copy}</p><Button kind="tertiary" size="sm" as={Link} to="/upload" renderIcon={Upload}>Upload video</Button></div>;
}

function DashboardPage() {
  const { stats, events, loading, error } = useDashboardData();
  return <Shell><main>
    <PageHeader title="Safety overview" description="Detection readiness, review workload, and recent evidence." action={<Button as={Link} to="/upload" renderIcon={Upload}>Upload video</Button>} />
    {error && <InlineNotification kind="error" title="Dashboard unavailable" subtitle={`${error}. Check the API connection and sign in again.`} lowContrast hideCloseButton />}
    {loading && <DashboardSkeleton />}
    {!loading && !error && stats && <>
    {!stats.model.available && <InlineNotification className="readiness-alert" kind="warning" title="Model not installed" subtitle="Complete governed training, lock best.pt, and install the verified weights before analysis." lowContrast hideCloseButton />}
    <MetricStrip stats={stats} />
    <div className="dashboard-grid">
      <section className="record-panel recent-panel"><div className="panel-heading"><div><h2>Recent events</h2><p>Latest reviewable detections</p></div><Button kind="ghost" size="sm" as={Link} to="/events">View all</Button></div><EventTable events={events.slice(0, 5)} compact /></section>
      <section className="record-panel queue-panel"><div className="panel-heading"><div><h2>Review queue</h2><p>Next item requiring inspection</p></div></div>{events.find((event) => ["PENDING", "NEEDS_REVIEW"].includes(event.review_status)) ? <QueueItem event={events.find((event) => ["PENDING", "NEEDS_REVIEW"].includes(event.review_status))!} /> : <EmptyState title="Queue is clear" copy="No events currently require review." />}</section>
      <section className="record-panel distribution"><div className="panel-heading"><div><h2>Event distribution</h2><p>Recorded system events, not model accuracy</p></div></div><Distribution stats={stats} /></section>
      <section className="record-panel provenance"><div className="panel-heading"><div><h2>Model provenance</h2><p>Current scientific artifact state</p></div></div><dl><div><dt>Model</dt><dd>{stats.model.model_version}</dd></div><div><dt>Weights</dt><dd>{stats.model.available ? "Installed" : "Not installed"}</dd></div><div><dt>Thresholds</dt><dd>{stats.model.threshold_status ?? "Awaiting validation"}</dd></div><div><dt>Fusion mode</dt><dd>{stats.runtime.fusion.fusion_mode}</dd></div><div><dt>Fusion gate</dt><dd>{stats.runtime.fusion.fusion_fail_closed ? "FAIL_CLOSED" : stats.runtime.fusion.fusion_available ? "ACTIVE" : stats.runtime.fusion.fusion_enabled ? "RULE_FALLBACK" : "DISABLED"}</dd></div><div><dt>Event metrics</dt><dd>{stats.event_metrics_status}</dd></div></dl></section>
    </div></>}
  </main></Shell>;
}

function DashboardSkeleton() { return <div className="dashboard-loading" aria-label="Loading dashboard"><div className="metric-loading" /><div className="panel-loading" /><div className="panel-loading" /></div>; }

function TableSkeleton() { return <div className="table-skeleton" aria-label="Loading events">{[1, 2, 3, 4].map((item) => <span key={item} />)}</div>; }

function QueueItem({ event }: { event: SafetyEvent }) { return <div className="queue-item"><div className="queue-preview"><Information size={38} /><span>Evidence available after analysis</span></div><dl><div><dt>Event type</dt><dd>{event.event_type.replace("_", " ")}</dd></div><div><dt>Occupant</dt><dd>{event.occupant_role.replaceAll("_", " ")}</dd></div><div><dt>Status</dt><dd><StatusTag status={event.review_status} /></dd></div><div><dt>Confidence</dt><dd>{Math.round(event.confidence * 100)}%</dd></div><div><dt>Frame</dt><dd>{event.frame_number}</dd></div></dl><Button as={Link} to={`/events/${event.id}`} size="sm">Review event</Button></div>; }

function Distribution({ stats }: { stats: Statistics }) {
  const items = [["Phone", stats.events_by_type.PHONE], ["No seatbelt", stats.events_by_type.NO_SEATBELT], ["Confirmed", stats.events_by_review_status.CONFIRMED], ["Pending review", stats.events_by_review_status.PENDING + stats.events_by_review_status.NEEDS_REVIEW]] as const;
  const max = Math.max(1, ...items.map(([, value]) => value));
  return <div className="distribution-bars">{items.map(([label, value]) => <div key={label}><span>{label}</span><i style={{ width: `${(value / max) * 100}%` }} /><strong>{value}</strong></div>)}</div>;
}

function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [inputScope, setInputScope] = useState<"VEHICLE_CABIN_CROP" | "TRAFFIC_SCENE_WITH_VEHICLE_ROIS">("VEHICLE_CABIN_CROP");
  const [state, setState] = useState<"idle" | "uploading" | "complete" | "error">("idle");
  const [message, setMessage] = useState("");
  async function start() {
    if (!file) return;
    setState("uploading");
    try { const video = await api.upload(file, inputScope); const job = await api.analyze(video.id); setMessage(`Analysis job ${job.id.slice(0, 8)} was queued.`); setState("complete"); }
    catch (reason) { setMessage(reason instanceof Error ? reason.message : "Upload failed"); setState("error"); }
  }
  return <Shell><main><PageHeader title="Upload and analyze" description="Submit one supported video for the locked model pipeline." />
    <section className="upload-layout"><div className="drop-sheet"><Select id="input-scope" labelText="Input scope" value={inputScope} onChange={(event) => setInputScope(event.target.value as typeof inputScope)}><SelectItem value="VEHICLE_CABIN_CROP" text="Validated vehicle cabin crop" /><SelectItem value="TRAFFIC_SCENE_WITH_VEHICLE_ROIS" text="Raw traffic scene (requires vehicle + cabin models)" /></Select><FileUploaderDropContainer accept={["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-matroska"]} labelText="Drop a video here or select a file" multiple={false} onAddFiles={(_, data) => setFile(data.addedFiles[0] ?? null)} /><p>MP4, MOV, AVI, MKV, or WebM. The server applies the configured size limit.</p>{file && <div className="selected-file"><Video size={24} /><span><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(1)} MB</small></span></div>}<Button onClick={start} disabled={!file || state === "uploading"} renderIcon={Play}>{state === "uploading" ? "Starting analysis" : "Analyze video"}</Button>{state === "uploading" && <InlineLoading description="Uploading and creating job" />}{state === "complete" && <InlineNotification kind="success" title="Analysis queued" subtitle={message} lowContrast hideCloseButton />}{state === "error" && <InlineNotification kind="error" title="Analysis could not start" subtitle={message} lowContrast hideCloseButton />}</div>
      <aside className="process-sheet"><h2>Before analysis</h2><ol><li><strong>Cabin context</strong><span>Raw traffic scenes fail closed unless a tracked vehicle has a confident windshield/cabin ROI.</span></li><li><strong>Locked V2 artifacts</strong><span>Specialist weights, calibrated thresholds, and required fusion artifacts must be installed.</span></li><li><strong>Seat geometry</strong><span>Camera handedness and cabin geometry determine occupant roles; ambiguous roles stay unknown.</span></li><li><strong>Human review</strong><span>Events remain candidates until an authorized reviewer checks the evidence.</span></li></ol></aside></section>
  </main></Shell>;
}

function EventsPage() {
  const [events, setEvents] = useState<SafetyEvent[]>([]); const [loading, setLoading] = useState(true); const [error, setError] = useState(""); const [type, setType] = useState(""); const [status, setStatus] = useState("");
  useEffect(() => { const query = new URLSearchParams(); if (type) query.set("event_type", type); if (status) query.set("review_status", status); api.events(`?${query}`).then(setEvents).catch((reason) => setError(reason.message)).finally(() => setLoading(false)); }, [type, status]);
  return <Shell><main><PageHeader title="Events" description="Search detections, inspect evidence, and record review decisions." action={<Button kind="tertiary" renderIcon={DocumentExport} href={api.exportUrl}>Export CSV</Button>} />
    <section className="filter-sheet"><Select id="event-type" labelText="Event type" value={type} onChange={(event) => setType(event.target.value)}><SelectItem value="" text="All types" /><SelectItem value="PHONE" text="Phone" /><SelectItem value="NO_SEATBELT" text="No seatbelt" /></Select><Select id="review-status" labelText="Review state" value={status} onChange={(event) => setStatus(event.target.value)}><SelectItem value="" text="All states" /><SelectItem value="PENDING" text="Pending" /><SelectItem value="NEEDS_REVIEW" text="Needs review" /><SelectItem value="CONFIRMED" text="Confirmed" /><SelectItem value="REJECTED" text="Rejected" /></Select></section>
    <section className="record-panel full-table"><div className="panel-heading"><div><h2>Event ledger</h2><p>{events.length} matching records</p></div></div>{error && <InlineNotification kind="error" title="Events unavailable" subtitle={error} lowContrast hideCloseButton />}{loading ? <TableSkeleton /> : <EventTable events={events} />}</section>
  </main></Shell>;
}

function EventDetailPage() {
  const { id = "" } = useParams();
  const [event, setEvent] = useState<EventDetail | null>(null);
  const [loadError, setLoadError] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const { assets, loading: evidenceLoading, error: evidenceError } = useEvidenceAssets(event);
  useEffect(() => { setLoadError(""); api.event(id).then(setEvent).catch((reason) => setLoadError(reason instanceof Error ? reason.message : "Event could not be loaded")); }, [id]);
  async function review(status: ReviewStatus) {
    if (!event || saving) return;
    setSaving(true); setReviewError("");
    try { await api.review(event.id, status, notes); setEvent(await api.event(event.id)); setNotes(""); }
    catch (reason) { setReviewError(reason instanceof Error ? reason.message : "Review decision could not be saved"); }
    finally { setSaving(false); }
  }
  return <Shell><main><PageHeader title="Event inspection" description="Compare the multi-frame evidence, model trace, and review provenance before deciding." />
    {loadError && <><InlineNotification kind="error" title="Event unavailable" subtitle={`${loadError}. Return to the event ledger and try again.`} lowContrast hideCloseButton /><Button className="recovery-action" kind="tertiary" as={Link} to="/events">Return to events</Button></>}
    {!event && !loadError ? <InlineLoading description="Loading event record" /> : event ? <>
      {event.review_status === "NEEDS_REVIEW" && <InlineNotification className="readiness-alert" kind="warning" title="Ambiguous evidence" subtitle="The runtime could not establish every required context confidently. Review the trace and preserve uncertainty when evidence is unclear." lowContrast hideCloseButton />}
      <div className="inspection-layout"><section className="evidence-sheet" aria-labelledby="evidence-heading"><div className="panel-heading"><div><h2 id="evidence-heading">Recorded evidence</h2><p>Protected files generated for this event</p></div></div>
        {evidenceLoading && <div className="evidence-loading"><InlineLoading description="Loading protected evidence" /></div>}
        {evidenceError && <InlineNotification kind="error" title="Evidence unavailable" subtitle={`${evidenceError}. The event record remains available; retry by reloading this page.`} lowContrast hideCloseButton />}
        {!evidenceLoading && !evidenceError && !event.evidence.available && <div className="evidence-placeholder"><Video size={48} /><strong>Evidence integrity incomplete</strong><span>{event.evidence.integrity_errors.join(" ") || "This event has no persisted evidence files."} Do not confirm it without independently verifiable evidence.</span></div>}
        {!evidenceLoading && assets.original && <div className="evidence-frames"><figure><img src={assets.original} alt="Original event key frame" /><figcaption>Original key frame</figcaption></figure>{assets.annotated && <figure><img src={assets.annotated} alt="Annotated event key frame showing the recorded detection" /><figcaption>Annotated key frame</figcaption></figure>}</div>}
        {!evidenceLoading && assets.clip && <div className="evidence-clip"><h3>Temporal evidence clip</h3><video controls preload="metadata" src={assets.clip}>Your browser cannot play this evidence clip.</video></div>}
        <div className="frame-meta"><span>Frame {event.frame_number}</span><span>{formatTimestamp(event.timestamp_seconds)}</span><span>Track {event.track_id ?? "Not assigned"}</span></div></section>
        <aside className="decision-sheet"><h2>Inspection record</h2><dl><div><dt>Event</dt><dd>{event.event_type.replace("_", " ")}</dd></div><div><dt>Occupant</dt><dd>{event.occupant_role.replaceAll("_", " ")}</dd></div><div><dt>Vehicle</dt><dd>{event.vehicle_type.replaceAll("_", " ")}</dd></div><div><dt>Vehicle context</dt><dd>{event.vehicle_context_id}</dd></div><div><dt>Event score</dt><dd>{Math.round(event.confidence * 100)}%</dd></div><div><dt>Fusion score</dt><dd>{event.fusion_score == null ? "NOT_AVAILABLE" : `${Math.round(event.fusion_score * 100)}%`}</dd></div><div><dt>Model</dt><dd>{event.model_version}</dd></div><div><dt>Current state</dt><dd><StatusTag status={event.review_status} /></dd></div></dl>
          {reviewError && <InlineNotification kind="error" title="Decision not saved" subtitle={reviewError} lowContrast hideCloseButton />}
          <TextArea id="review-notes" labelText="Review notes" helperText={`${notes.length}/2000 characters. Record visible evidence and ambiguity; never infer an unseen belt state.`} maxLength={2000} value={notes} onChange={(event) => setNotes(event.target.value)} />
          <div className="decision-actions"><Button size="sm" disabled={saving || !event.evidence.confirmation_ready} onClick={() => review("CONFIRMED")}>{saving ? "Saving" : "Confirm"}</Button><Button size="sm" kind="secondary" disabled={saving} onClick={() => review("NEEDS_REVIEW")}>Needs review</Button><Button size="sm" kind="danger--tertiary" disabled={saving} onClick={() => review("REJECTED")}>Reject</Button></div>
          {!event.evidence.confirmation_ready && <p className="decision-guard">Confirmation is disabled: {event.evidence.confirmation_blockers.join(" ") || "complete temporal evidence is unavailable."}</p>}
        </aside></div>
      <section className="audit-grid"><div className="record-panel trace-panel"><div className="panel-heading"><div><h2>Decision trace</h2><p>Context and temporal evidence recorded at event creation</p></div></div><dl>{Object.entries(event.evidence_trace).filter(([key]) => ["vehicle_track_id", "cabin_bbox", "cabin_method", "cabin_confidence", "behavior_track_id", "occupant_association_method", "occupant_role_confidence", "phone_context", "pose_confidence", "fusion_mode", "temporal_score", "evidence_source", "evidence_status"].includes(key)).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{typeof value === "number" ? value.toFixed(3) : String(value)}</dd></div>)}</dl></div>
        <div className="record-panel history-panel"><div className="panel-heading"><div><h2>Review history</h2><p>Append-only reviewer provenance</p></div></div>{event.review_history.length ? <ol>{event.review_history.map((record, index) => <li key={`${record.created_at}-${index}`}><div><strong>{record.previous_status} → {record.new_status}</strong><time dateTime={record.created_at}>{new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(record.created_at))}</time></div><p>{record.notes || "No notes recorded."}</p><small>Reviewer {record.reviewer_id}</small></li>)}</ol> : <div className="history-empty">No human review decision has been recorded.</div>}</div></section>
    </> : null}</main></Shell>;
}

function StatisticsPage() { const { stats, loading, error } = useDashboardData(); return <Shell><main><PageHeader title="Statistics" description="Operational counts are separated from scientific model and event metrics." />{error && <InlineNotification kind="error" title="Statistics unavailable" subtitle={error} lowContrast hideCloseButton />}{loading ? <InlineLoading description="Loading statistics" /> : !error && stats ? <><MetricStrip stats={stats} /><div className="statistics-grid"><section className="record-panel"><h2>Recorded events</h2><Distribution stats={stats} /></section><section className="record-panel"><h2>Metric status</h2><dl className="metric-status"><div><dt>Detection validation</dt><dd>NOT_RUN</dd></div><div><dt>Frozen test</dt><dd>NOT_RUN</dd></div><div><dt>Human event evaluation</dt><dd>{stats.event_metrics_status}</dd></div><div><dt>Fusion runtime</dt><dd>{stats.runtime.fusion.fusion_mode}{stats.runtime.fusion.fusion_fail_closed ? " / FAIL_CLOSED" : stats.runtime.fusion.fusion_available ? " / ACTIVE" : ""}</dd></div></dl><p className="scientific-note">Event counts are not accuracy. Metrics appear only after governed evaluation artifacts exist.</p></section></div></> : null}</main></Shell>; }

function ModelsPage() { return <Shell><main><PageHeader title="Models and methodology" description="The immutable V1 baseline and independent fail-closed V2 pipeline." /><section className="model-sheet"><div><h2>MC_BOOTSTRAP_001 — V1 baseline</h2><p>Single YOLO11s detector retained unchanged to measure partial-label conflict.</p></div><dl><div><dt>Classes</dt><dd>phone, seatbelt_fastened, seatbelt_unfastened</dd></div><div><dt>Training</dt><dd>150 epochs; scientific baseline only</dd></div><div><dt>V2 architecture</dt><dd>Vehicle → cabin → occupant → phone/seatbelt specialists → temporal fusion</dd></div><div><dt>Current V2 state</dt><dd><Tag type="warm-gray">UNTRAINED / NOT APPROVED</Tag></dd></div></dl><InlineNotification kind="info" title="Detection is not behavior" subtitle="A candidate requires confident vehicle and cabin context, occupant association, temporal persistence, explicit fusion mode, and human review. Component AP is not event accuracy." lowContrast hideCloseButton /></section></main></Shell>; }

function SettingsPage() { return <Shell><main><PageHeader title="Settings" description="Camera geometry and temporal rules are deployment configuration." /><section className="settings-sheet"><Select id="driver-side" labelText="Driver side" defaultValue="left"><SelectItem value="left" text="Left-hand drive" /><SelectItem value="right" text="Right-hand drive" /></Select><TextInput id="window-seconds" labelText="Confirmation window (seconds)" defaultValue="2.0" helperText="Tune on validation video only." /><TextInput id="positive-seconds" labelText="Minimum positive duration (seconds)" defaultValue="1.2" helperText="Frozen test video must not be used for tuning." /><InlineNotification kind="warning" title="Configuration preview" subtitle="This screen documents expected settings. Persist camera-specific changes through the authenticated API before production use." lowContrast hideCloseButton /><Button disabled>Save deployment settings</Button></section></main></Shell>; }

function ProtectedApp() {
  if (!sessionStorage.getItem("roadwatch-token")) return <Navigate to="/login" replace />;
  return <Routes><Route path="/" element={<DashboardPage />} /><Route path="/upload" element={<UploadPage />} /><Route path="/events" element={<EventsPage />} /><Route path="/events/:id" element={<EventDetailPage />} /><Route path="/statistics" element={<StatisticsPage />} /><Route path="/models" element={<ModelsPage />} /><Route path="/settings" element={<SettingsPage />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes>;
}

export default function App() { return <Routes><Route path="/login" element={<LoginPage />} /><Route path="/*" element={<ProtectedApp />} /></Routes>; }
