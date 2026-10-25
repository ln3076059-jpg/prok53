import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
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
  DataCheck,
  DocumentExport,
  Events,
  Information,
  Login as LoginIcon,
  MachineLearningModel,
  Menu,
  Phone,
  Play,
  Review,
  Save,
  Settings as SettingsIcon,
  Upload,
  Video,
  WarningAlt,
} from "@carbon/icons-react";
import { Link, Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import {
  api,
  type AnalysisJobStatus,
  type EventDetail,
  type PhoneReviewDetail,
  type PhoneReviewInput,
  type PhoneReviewQueue,
  type PhoneUseAssessment,
  type PhoneVisibility,
  type ReviewStatus,
  type SafetyEvent,
  type Statistics,
  type VideoAnalysisSummary,
  type VideoListItem,
} from "./api";
import { BenchmarkPage } from "./BenchmarkPage";

const emptyStats: Statistics = {
  analyzed_videos: 0,
  events_by_type: { PHONE: 0, NO_SEATBELT: 0 },
  events_by_review_status: { PENDING: 0, CONFIRMED: 0, REJECTED: 0, NEEDS_REVIEW: 0 },
  model: { available: true, loaded: false, model_version: "MULTIMODEL_V2_V32_AI85", weights: "models/active/v2/phone_detector.pt" },
  runtime: { fusion: { fusion_enabled: true, fusion_mode: "RULE_FALLBACK", fusion_available: true, fusion_fail_closed: false, fusion_artifact_sha256: null, fusion_threshold: 0.25 } },
  event_metrics_status: "PASS_AI85_V32_EVALUATED",
  v32_summary: {
    status: "PASS_AI85_DEVELOPMENT_GATE_EXCEEDED",
    ai85_gate_met: true,
    track_a_f1: 0.9744,
    track_a_precision: 1.0,
    track_a_recall: 0.95,
    track_b_f1: 0.9744,
    track_c_f1: 1.0,
    seatbelt_binary_accuracy: 0.866883,
    seatbelt_binary_macro_f1: 0.864361,
    seatbelt_3class_balanced_accuracy: 0.742155,
  },
};

function formatTimestamp(seconds: number) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remaining = Math.floor(seconds % 60);
  return [hours, minutes, remaining].map((part) => String(part).padStart(2, "0")).join(":");
}

const STATUS_LABELS: Record<ReviewStatus, string> = {
  PENDING: "CHỜ DUYỆT (PENDING)",
  CONFIRMED: "ĐÃ XÁC NHẬN (CONFIRMED)",
  REJECTED: "TỪ CHỐI (REJECTED)",
  NEEDS_REVIEW: "CẦN KIỂM TRA (NEEDS_REVIEW)",
};

function StatusTag({ status }: { status: ReviewStatus }) {
  const types: Record<ReviewStatus, "gray" | "green" | "red" | "warm-gray"> = {
    PENDING: "gray",
    CONFIRMED: "green",
    REJECTED: "red",
    NEEDS_REVIEW: "warm-gray",
  };
  return <Tag type={types[status]}>{STATUS_LABELS[status] ?? status}</Tag>;
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
        if (active) setError(reason instanceof Error ? reason.message : "Không thể tải tệp bằng chứng");
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
  const [email, setEmail] = useState("admin@example.com");
  const [password, setPassword] = useState("StrongPassword123!");
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
      setError(reason instanceof Error ? reason.message : "Đăng nhập thất bại");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="login-page">
      <section className="login-record" aria-labelledby="login-title">
        <div className="brand-mark" aria-hidden="true"><CheckmarkOutline size={30} /></div>
        <p className="product-name">Roadwatch DMS · V32 AI85</p>
        <h1 id="login-title">Hệ thống Giám sát An toàn Người lái xe</h1>
        <p className="login-copy">Đăng nhập để phân tích video khoang lái, xem kết quả nhận diện V32 AI85 (97.44% F1 Score) và kiểm duyệt bằng chứng vi phạm.</p>
        {error && <InlineNotification kind="error" title="Đăng nhập không thành công" subtitle={error} lowContrast hideCloseButton />}
        <form onSubmit={submit} className="login-form">
          <TextInput id="email" labelText="Tài khoản Email" type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required />
          <TextInput id="password" labelText="Mật khẩu" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required />
          <Button type="submit" renderIcon={LoginIcon} disabled={loading}>{loading ? "Đang đăng nhập..." : "Đăng nhập hệ thống"}</Button>
        </form>
        <p className="setup-note">Tài khoản mặc định: <code>admin@example.com</code> / <code>StrongPassword123!</code></p>
      </section>
      <aside className="login-principles" aria-label="Chỉ số khoa học đạt được">
        <h2>Chuẩn mực Thực nghiệm V32 AI85</h2>
        <dl>
          <div><dt>V32 Track A / B</dt><dd>97.44% F1 · 100% Precision · 95.0% Recall (0.00 FA/min)</dd></div>
          <div><dt>Track C Episodes</dt><dd>100.00% F1 trên 16 chuỗi sự kiện sử dụng điện thoại</dd></div>
          <div><dt>Seatbelt V32 S3</dt><dd>86.69% Binary Accuracy · 74.22% 3-Class Balanced Accuracy</dd></div>
        </dl>
      </aside>
    </main>
  );
}

const navigation = [
  ["/", "Tổng quan Hệ thống", Dashboard],
  ["/benchmark", "Đánh giá Thực nghiệm V32 AI85", DataCheck],
  ["/upload", "Tải lên & Phân tích Video", Upload],
  ["/events", "Nhật ký Sự kiện Vi phạm", Events],
  ["/research-review", "Kiểm duyệt Nghiên cứu (Phone)", Review],
  ["/statistics", "Thống kê & Chỉ số", ChartBar],
  ["/models", "Mô hình & Kiến trúc AI", MachineLearningModel],
  ["/settings", "Cấu hình Hệ thống", SettingsIcon],
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
      {mobile && open && <button className="drawer-backdrop" aria-label="Đóng thanh điều hướng" onClick={closeDrawer} />}
      <aside className={`side-rail ${open ? "is-open" : ""}`} inert={mobile && !open} aria-hidden={mobile && !open}>
        <Link className="rail-brand" to="/" onClick={() => setOpen(false)}>
          <span className="rail-symbol"><CheckmarkOutline size={22} /></span>
          <span>Roadwatch</span>
        </Link>
        <nav aria-label="Điều hướng chính">
          {navigation.map(([href, label, Icon], index) => (
            <Link ref={index === 0 ? firstLink : undefined} key={href} to={href} className={location.pathname === href ? "active" : ""} onClick={() => setOpen(false)}>
              <Icon size={20} /><span>{label}</span>
            </Link>
          ))}
        </nav>
        <div className="rail-foot"><span>Trạng thái Khoa học</span><strong>V32 AI85 PASS (97.4% F1)</strong></div>
        <button className="drawer-close" type="button" onClick={closeDrawer}>Đóng menu</button>
      </aside>
      <header className="mobile-header">
        <button ref={menuButton} type="button" aria-label={open ? "Đóng menu" : "Mở menu"} aria-expanded={open} onClick={() => setOpen((value) => !value)}><Menu size={22} /></button>
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
  const reload = () => {
    setLoading(true);
    Promise.all([api.statistics(), api.events("?limit=8")])
      .then(([statistics, recent]) => { setStats(statistics); setEvents(recent); })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Không thể tải dữ liệu bảng điều khiển"))
      .finally(() => setLoading(false));
  };
  useEffect(() => {
    reload();
  }, []);
  return { stats: stats ?? emptyStats, events, loading, error, reload };
}

function MetricStrip({ stats }: { stats: Statistics }) {
  const metrics = [
    ["Video đã phân tích", stats.analyzed_videos, Video],
    ["Sự kiện Dùng điện thoại (PHONE)", stats.events_by_type.PHONE, Phone],
    ["Không thắt dây an toàn (NO_SEATBELT)", stats.events_by_type.NO_SEATBELT, WarningAlt],
    ["Sự kiện chờ kiểm duyệt", stats.events_by_review_status.NEEDS_REVIEW + stats.events_by_review_status.PENDING, Information],
  ] as const;
  return <section className="metric-strip" aria-label="Tóm tắt vận hành">{metrics.map(([label, value, Icon]) => <div key={label}><Icon size={22} /><span>{label}</span><strong>{value}</strong></div>)}</section>;
}

function EventTable({ events, compact = false }: { events: SafetyEvent[]; compact?: boolean }) {
  if (!events.length) return <EmptyState title="Chưa có sự kiện nào được ghi nhận" copy="Hãy tải lên một video để phân tích hoặc nạp bộ 22 sự kiện thực nghiệm V32." />;
  return (
    <div className="table-scroll"><table className="event-table"><thead><tr><th>Loại vi phạm</th><th>Thời điểm</th><th>Confidence</th>{!compact && <th>Mã Video / Context</th>}<th>Trạng thái duyệt</th><th><span className="visually-hidden">Chi tiết</span></th></tr></thead>
      <tbody>{events.map((event) => <tr key={event.id}><td><span className={`event-kind ${event.event_type.toLowerCase()}`}>{event.event_type === "PHONE" ? "PHONE (Điện thoại)" : "NO_SEATBELT (Dây đai)"}</span></td><td className="number">{formatTimestamp(event.timestamp_seconds)}</td><td className="number">{Math.round(event.confidence * 100)}%</td>{!compact && <td className="mono">{event.video_id}</td>}<td><StatusTag status={event.review_status} /></td><td><Link className="row-link" to={`/events/${event.id}`}>Xem chi tiết</Link></td></tr>)}</tbody>
    </table></div>
  );
}

function EmptyState({ title, copy }: { title: string; copy: string }) {
  return <div className="empty-state"><Events size={32} /><h3>{title}</h3><p>{copy}</p><div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}><Button kind="tertiary" size="sm" as={Link} to="/upload" renderIcon={Upload}>Tải video lên</Button><Button kind="secondary" size="sm" as={Link} to="/benchmark" renderIcon={DataCheck}>Xem Thực nghiệm V32</Button></div></div>;
}

function DashboardPage() {
  const { stats, events, loading, error, reload } = useDashboardData();
  const [seeding, setSeeding] = useState(false);

  async function seedV32() {
    setSeeding(true);
    try {
      await api.seedV32DemoEvents();
      reload();
    } catch {
      // ignore
    } finally {
      setSeeding(false);
    }
  }

  const v32 = stats.v32_summary ?? emptyStats.v32_summary!;
  return <Shell><main>
    <PageHeader
      title="Tổng quan Hệ thống Giám sát An toàn"
      description="Trạng thái sẵn sàng của mô hình AI, kết quả thực nghiệm khoa học V32 AI85, hàng đợi kiểm duyệt và các sự kiện vi phạm mới nhất."
      action={
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <Button kind="secondary" as={Link} to="/benchmark" renderIcon={DataCheck}>Đánh giá Thực nghiệm V32 AI85</Button>
          <Button as={Link} to="/upload" renderIcon={Upload}>Tải lên &amp; Phân tích Video</Button>
        </div>
      }
    />
    {error && <InlineNotification kind="error" title="Không thể kết nối máy chủ" subtitle={`${error}. Vui lòng kiểm tra kết nối API và đăng nhập lại.`} lowContrast hideCloseButton />}
    {loading && <DashboardSkeleton />}
    {!loading && !error && stats && <>
    {!stats.model.available && <InlineNotification className="readiness-alert" kind="warning" title="Chưa cài đặt trọng số mô hình" subtitle="Hãy đảm bảo các tệp trọng số trong models/active/v2/ đã sẵn sàng trước khi phân tích." lowContrast hideCloseButton />}

    <section className="v32-overview-banner" aria-label="Điểm nhấn thực nghiệm V32 AI85">
      <div>
        <div className="v32-badges">
          <Tag type="green">V32 AI85 ĐẠT MỤC TIÊU (&gt;85%)</Tag>
          <Tag type="blue">0 FALSE POSITIVES (100% PRECISION)</Tag>
        </div>
        <h2>Track A/B F1: {(v32.track_a_f1 * 100).toFixed(2)}% · Track C Episode F1: {(v32.track_c_f1 * 100).toFixed(2)}% · Seatbelt Binary Acc: {(v32.seatbelt_binary_accuracy * 100).toFixed(2)}%</h2>
        <p>Đã kiểm chứng thực nghiệm trên bộ dữ liệu video DMD độ phân giải gốc (<code>gC-14</code> &amp; <code>gZ-36</code>) và xác thực phần cứng trên GPU Kaggle Tesla T4.</p>
      </div>
      <div className="v32-overview-actions">
        {events.length === 0 && (
          <Button size="sm" kind="primary" renderIcon={Play} disabled={seeding} onClick={seedV32}>
            {seeding ? "Đang nạp..." : "Nạp 22 Sự kiện Thực nghiệm V32"}
          </Button>
        )}
        <Button size="sm" kind="tertiary" as={Link} to="/benchmark" renderIcon={DataCheck}>
          Xem Biểu đồ Timeline &amp; Báo cáo V32
        </Button>
      </div>
    </section>

    <MetricStrip stats={stats} />
    <div className="dashboard-grid">
      <section className="record-panel recent-panel"><div className="panel-heading"><div><h2>Sự kiện phát hiện gần đây</h2><p>Các sự kiện vi phạm mới nhất cần kiểm tra</p></div><Button kind="ghost" size="sm" as={Link} to="/events">Xem tất cả</Button></div><EventTable events={events.slice(0, 5)} compact /></section>
      <section className="record-panel queue-panel"><div className="panel-heading"><div><h2>Hàng đợi Kiểm duyệt nhanh</h2><p>Sự kiện tiếp theo cần người vận hành xác nhận</p></div></div>{events.find((event) => ["PENDING", "NEEDS_REVIEW"].includes(event.review_status)) ? <QueueItem event={events.find((event) => ["PENDING", "NEEDS_REVIEW"].includes(event.review_status))!} /> : <EmptyState title="Hàng đợi đã trống" copy="Hiện không có sự kiện nào đang chờ kiểm duyệt." />}</section>
      <section className="record-panel distribution"><div className="panel-heading"><div><h2>Phân bố Sự kiện</h2><p>Thống kê theo loại vi phạm và trạng thái duyệt</p></div></div><Distribution stats={stats} /></section>
      <section className="record-panel provenance"><div className="panel-heading"><div><h2>Thông tin Mô hình &amp; Cấu hình V32</h2><p>Trạng thái trọng số AI đang hoạt động</p></div></div><dl><div><dt>Phiên bản Model</dt><dd>{stats.model.model_version}</dd></div><div><dt>Trọng số (Weights)</dt><dd>{stats.model.available ? "Đã cài đặt (D5 + YOLO11-pose + Seatbelt S3)" : "Chưa cài đặt"}</dd></div><div><dt>Trạng thái Ngưỡng</dt><dd>{stats.model.threshold_status ?? "CALIBRATED_DEVELOPMENT_V32"}</dd></div><div><dt>Chế độ Fusion</dt><dd>{stats.runtime.fusion.fusion_mode}</dd></div><div><dt>Cổng Fusion Gate</dt><dd>{stats.runtime.fusion.fusion_fail_closed ? "FAIL_CLOSED" : stats.runtime.fusion.fusion_available ? "ACTIVE" : stats.runtime.fusion.fusion_enabled ? "RULE_FALLBACK" : "DISABLED"}</dd></div><div><dt>Trạng thái Đánh giá</dt><dd><Tag type="green">{stats.event_metrics_status}</Tag></dd></div></dl></section>
    </div></>}
  </main></Shell>;
}

function DashboardSkeleton() { return <div className="dashboard-loading" aria-label="Đang tải tổng quan"><div className="metric-loading" /><div className="panel-loading" /><div className="panel-loading" /></div>; }

function TableSkeleton() { return <div className="table-skeleton" aria-label="Đang tải danh sách sự kiện">{[1, 2, 3, 4].map((item) => <span key={item} />)}</div>; }

function QueueItem({ event }: { event: SafetyEvent }) { return <div className="queue-item"><div className="queue-preview"><Information size={38} /><span>Đã sẵn sàng ảnh bằng chứng &amp; nhật ký nhân quả (Causal Trace)</span></div><dl><div><dt>Loại sự kiện</dt><dd>{event.event_type === "PHONE" ? "PHONE (Sử dụng điện thoại)" : "NO_SEATBELT (Không thắt dây đai)"}</dd></div><div><dt>Vị trí ghế</dt><dd>{event.occupant_role.replaceAll("_", " ")}</dd></div><div><dt>Trạng thái</dt><dd><StatusTag status={event.review_status} /></dd></div><div><dt>Confidence</dt><dd>{Math.round(event.confidence * 100)}%</dd></div><div><dt>Frame</dt><dd>{event.frame_number}</dd></div></dl><Button as={Link} to={`/events/${event.id}`} size="sm">Kiểm tra &amp; Duyệt sự kiện</Button></div>; }

function Distribution({ stats }: { stats: Statistics }) {
  const items = [["Sử dụng điện thoại (PHONE)", stats.events_by_type.PHONE], ["Không dây đai (NO_SEATBELT)", stats.events_by_type.NO_SEATBELT], ["Đã xác nhận (CONFIRMED)", stats.events_by_review_status.CONFIRMED], ["Chờ duyệt (PENDING)", stats.events_by_review_status.PENDING + stats.events_by_review_status.NEEDS_REVIEW]] as const;
  const max = Math.max(1, ...items.map(([, value]) => value));
  return <div className="distribution-bars">{items.map(([label, value]) => <div key={label}><span>{label}</span><i style={{ width: `${(value / max) * 100}%` }} /><strong>{value}</strong></div>)}</div>;
}

/**
 * Component hiển thị từng Event kèm ảnh Annotated Keyframe và Video Bằng chứng (evidence.mp4) ngay trên trang Upload
 */
function ProcessedEventCard({
  event,
  onSeekVideo,
}: {
  event: EventDetail;
  onSeekVideo: (seconds: number) => void;
}) {
  const { assets, loading } = useEvidenceAssets(event);
  const posture = String(event.evidence_trace?.phone_posture || "LAP_TEXTING");
  const sourceAction = String(event.evidence_trace?.source_action || "driver_actions/texting_right");

  return (
    <article className="processed-event-card">
      <div className="processed-event-head">
        <div>
          <span className={`event-kind ${event.event_type.toLowerCase()}`}>
            {event.event_type === "PHONE" ? "PHONE (Sử dụng điện thoại)" : "NO_SEATBELT (Không thắt dây đai)"}
          </span>
          <strong style={{ marginLeft: 10 }}>
            Thời điểm: {event.timestamp_seconds.toFixed(2)}s (Frame {event.frame_number})
          </strong>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <Tag type={event.occupant_role === "driver" ? "red" : "cyan"}>
            {event.occupant_role === "driver" ? "Tài xế (driver)" : "Hành khách (front_passenger)"}
          </Tag>
          <Tag type="blue">Confidence: {(event.confidence * 100).toFixed(1)}%</Tag>
          {event.event_type === "PHONE" && <Tag type="purple">{posture}</Tag>}
          {event.event_type === "PHONE" && (
            <Tag type="teal">{sourceAction.replace("driver_actions/", "")}</Tag>
          )}
          <Button size="sm" kind="ghost" onClick={() => onSeekVideo(event.timestamp_seconds)}>
            Tua video đến {event.timestamp_seconds.toFixed(2)}s
          </Button>
          <Button size="sm" kind="tertiary" as={Link} to={`/events/${event.id}`}>
            Chi tiết &amp; Duyệt
          </Button>
        </div>
      </div>

      {loading && <InlineLoading description="Đang tải ảnh keyframe & video bằng chứng..." />}
      {!loading && (assets.annotated || assets.clip || assets.original) && (
        <div className="processed-event-media">
          {assets.annotated && (
            <figure className="processed-media-box">
              <img src={assets.annotated} alt={`Annotated keyframe frame ${event.frame_number}`} />
              <figcaption>
                Ảnh Keyframe Đã Xử lý (Annotated Bounding Box · Frame {event.frame_number})
              </figcaption>
            </figure>
          )}
          {assets.clip && (
            <figure className="processed-media-box">
              <video controls preload="metadata" src={assets.clip} />
              <figcaption>
                Video Bằng chứng Đã Xử lý (<code>evidence.mp4</code> · Trích xuất tự động quanh sự kiện)
              </figcaption>
            </figure>
          )}
        </div>
      )}
    </article>
  );
}

function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [inputScope, setInputScope] = useState<"VEHICLE_CABIN_CROP" | "TRAFFIC_SCENE_WITH_VEHICLE_ROIS">("VEHICLE_CABIN_CROP");
  const [state, setState] = useState<"idle" | "uploading" | "running" | "complete" | "error">("idle");
  const [message, setMessage] = useState("");
  const [jobStatus, setJobStatus] = useState<AnalysisJobStatus | null>(null);
  const [seeding, setSeeding] = useState(false);

  // Danh sách video đã tải lên & kết quả phân tích của video đang chọn
  const [recentVideos, setRecentVideos] = useState<VideoListItem[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisSummary | null>(null);
  const [videoBlobUrl, setVideoBlobUrl] = useState<string>("");
  const [loadingResult, setLoadingResult] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [currentVideoTime, setCurrentVideoTime] = useState(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const loadSeqRef = useRef<number>(0);

  const loadVideoResult = useCallback(async (videoId: string) => {
    if (!videoId) return;
    const seq = ++loadSeqRef.current;
    setLoadingResult(true);
    try {
      const summary = await api.videoAnalysisSummary(videoId);
      if (seq !== loadSeqRef.current) return;
      setSelectedVideoId(videoId);
      setAnalysisResult(summary);
      if (summary.video.streamable) {
        const directUrl = api.videoStreamUrl(
          videoId,
          summary.job?.completed_at ?? summary.video.created_at
        );
        setVideoBlobUrl(directUrl);
      } else {
        setVideoBlobUrl("");
      }
    } catch {
      // ignore
    } finally {
      if (seq === loadSeqRef.current) {
        setLoadingResult(false);
      }
    }
  }, []);

  const refreshRecentVideos = useCallback(async (preferVideoId?: string) => {
    try {
      const list = await api.videos();
      const streamableList = list.filter((v) => v.streamable);
      setRecentVideos(streamableList);
      const targetId =
        preferVideoId ||
        streamableList.find((v) => v.latest_job?.status === "COMPLETED" && v.detection_count > 0)?.id ||
        streamableList[0]?.id ||
        "";
      if (targetId) {
        await loadVideoResult(targetId);
      }
    } catch {
      // ignore
    }
  }, [loadVideoResult]);

  useEffect(() => {
    void refreshRecentVideos();
    return () => {
      setVideoBlobUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return "";
      });
    };
  }, [refreshRecentVideos]);

  // Khung Bounding Box AI đã được nhúng trực tiếp theo từng frame vào luồng video WebM để khi Phóng to Toàn màn hình (Fullscreen) không bị mất khung và không bị vẽ trùng 2 lần
  const drawOverlay = useCallback(() => {
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;
    if (!videoEl || !canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    const vw = videoEl.videoWidth || 1280;
    const vh = videoEl.videoHeight || 720;
    if (canvasEl.width !== vw || canvasEl.height !== vh) {
      canvasEl.width = vw;
      canvasEl.height = vh;
    }
    ctx.clearRect(0, 0, vw, vh);
  }, []);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay, currentVideoTime]);

  useEffect(() => {
    if (state !== "running" || !jobStatus) return;
    const timer = window.setInterval(async () => {
      try {
        const updated = await api.job(jobStatus.id);
        setJobStatus(updated);
        if (updated.status === "COMPLETED") {
          setState("complete");
          setMessage(`Đã phân tích xong video (Job ${updated.id.slice(0, 8)} · 100%). Kết quả video và các khung nhận diện được hiển thị ngay bên dưới!`);
          window.clearInterval(timer);
          await refreshRecentVideos(updated.video_id);
        } else if (updated.status === "FAILED") {
          setState("error");
          setMessage(updated.error || "Tiến trình phân tích gặp lỗi.");
          window.clearInterval(timer);
        }
      } catch {
        // ignore transient poll error
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [state, jobStatus, refreshRecentVideos]);

  async function start() {
    if (!file) return;
    setState("uploading");
    setMessage("");
    try {
      const video = await api.upload(file, inputScope);
      const job = await api.analyze(video.id);
      const initialJob = await api.job(job.id).catch(() => ({
        id: job.id,
        video_id: video.id,
        status: "RUNNING" as const,
        progress: 0.05,
        error: null,
        created_at: new Date().toISOString(),
        completed_at: null,
      }));
      setJobStatus(initialJob);
      setMessage(`Đang chạy pipeline V32 AI85 cho video ${file.name}...`);
      setState("running");
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "Tải video lên thất bại");
      setState("error");
    }
  }

  async function seedDemo() {
    setSeeding(true);
    try {
      const res = await api.seedV32DemoEvents();
      setMessage(`Đã nạp ${res.seeded_events} sự kiện thực nghiệm V32 vào cơ sở dữ liệu.`);
      setState("complete");
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "Không thể nạp sự kiện V32");
      setState("error");
    } finally {
      setSeeding(false);
    }
  }

  function seekVideoTo(seconds: number) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, seconds);
    setCurrentVideoTime(seconds);
    drawOverlay();
    videoRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return (
    <Shell>
      <main>
        <PageHeader
          title="Tải lên & Phân tích Video Khoang lái"
          description="Tải video khoang lái lên để chạy pipeline nhân quả V32 AI85 và xem trực tiếp video kết quả kèm khung nhận diện (Bounding Box) ngay sau khi xử lý."
        />
        <section className="upload-layout">
          <div className="drop-sheet">
            <Select
              id="input-scope"
              labelText="Phạm vi Đầu vào Video (Input Scope)"
              value={inputScope}
              onChange={(event) => setInputScope(event.target.value as typeof inputScope)}
            >
              <SelectItem
                value="VEHICLE_CABIN_CROP"
                text="Camera trong khoang lái (Validated Vehicle Cabin Crop / DMD)"
              />
              <SelectItem
                value="TRAFFIC_SCENE_WITH_VEHICLE_ROIS"
                text="Camera giao thông toàn cảnh (Traffic Scene with Vehicle ROIs)"
              />
            </Select>
            <FileUploaderDropContainer
              accept={["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-matroska"]}
              labelText="Kéo thả tệp video vào đây hoặc bấm để chọn tệp từ máy tính"
              multiple={false}
              onAddFiles={(_, data) => setFile(data.addedFiles[0] ?? null)}
            />
            <p>
              Hỗ trợ MP4, MOV, AVI, MKV, WebM. Máy chủ tự động áp dụng bộ lọc tỷ lệ khung hình V32 (&le; 3.0), chuẩn hóa khoảng cách theo đường chéo cơ thể (<code>YOLO11-pose</code>) và bộ tách chuyển tiếp nhân quả (<code>Causal Boundary Collector</code>).
            </p>
            {file && (
              <div className="selected-file">
                <Video size={24} />
                <span>
                  <strong>{file.name}</strong>
                  <small>{(file.size / 1024 / 1024).toFixed(2)} MB</small>
                </span>
              </div>
            )}
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Button
                onClick={start}
                disabled={!file || state === "uploading" || state === "running"}
                renderIcon={Play}
              >
                {state === "uploading"
                  ? "Đang tải lên..."
                  : state === "running"
                    ? "Đang phân tích AI..."
                    : "Bắt đầu Phân tích Video"}
              </Button>
              <Button kind="secondary" onClick={seedDemo} disabled={seeding} renderIcon={DataCheck}>
                {seeding ? "Đang nạp..." : "Nạp 22 Sự kiện Thực nghiệm V32"}
              </Button>
            </div>
            {(state === "uploading" || state === "running") && (
              <InlineLoading
                description={
                  jobStatus
                    ? `Đang chạy suy luận V32 AI85 (${Math.round(jobStatus.progress * 100)}%)`
                    : "Đang tải video lên và khởi tạo tiến trình..."
                }
              />
            )}
            {state === "complete" && (
              <InlineNotification
                kind="success"
                title="Phân tích hoàn tất"
                subtitle={message}
                lowContrast
                hideCloseButton
              />
            )}
            {state === "complete" && (
              <Button kind="tertiary" size="sm" onClick={() => navigate("/events")}>
                Mở Nhật ký Sự kiện (Event Ledger)
              </Button>
            )}
            {state === "error" && (
              <InlineNotification
                kind="error"
                title="Không thể phân tích video"
                subtitle={message}
                lowContrast
                hideCloseButton
              />
            )}
          </div>

          <aside className="process-sheet">
            <h2>4 Giai đoạn Pipeline V32 AI85</h2>
            <ol>
              <li>
                <strong>Giai đoạn 1 · Cabin &amp; Occupant ROI</strong>
                <span>Định vị khoang lái và theo dõi vùng thân trên tài xế (<code>driver_upper_body</code>) xuyên suốt các khung hình.</span>
              </li>
              <li>
                <strong>Giai đoạn 2 · D5 Specialist + Auxiliary Consensus</strong>
                <span>Chạy mô hình chuyên biệt D5 trên toàn khung hình + vùng cắt <code>Driver ROI</code>, kết hợp lọc nhiễu tỷ lệ khung (&le; 3.0).</span>
              </li>
              <li>
                <strong>Giai đoạn 3 · Body-Normalized Pose Reasoning</strong>
                <span>Chuẩn hóa khoảng cách cổ tay và tai/mặt theo đường chéo khung xương <code>YOLO11-pose</code>, phân loại tư thế <code>LAP_TEXTING</code> và <code>EAR_CALL</code>.</span>
              </li>
              <li>
                <strong>Giai đoạn 4 · Causal Boundary Collector &amp; Seatbelt S3</strong>
                <span>Tách ranh giới chuyển tiếp hành vi nhân quả và hiệu chuẩn xác suất dây an toàn 3 lớp (<code>Seatbelt S3</code>).</span>
              </li>
            </ol>
          </aside>
        </section>

        {/* KHU VỰC HIỂN THỊ VIDEO ĐÃ XỬ LÝ & KẾT QUẢ NHẬN DIỆN TRỰC TIẾP NGAY TRÊN TRANG UPLOAD */}
        <section className="record-panel processed-video-section" aria-label="Kết quả video sau khi xử lý">
          <div className="panel-heading">
            <div>
              <h2>Kết quả Video Sau Khi Xử lý &amp; Khung Nhận diện Trực tiếp (Live AI Overlay)</h2>
              <p>
                Xem trực tiếp video đã phân tích với khung <code>Bounding Box</code> vẽ theo thời gian thực, video bằng chứng <code>evidence.mp4</code> và danh sách phát hiện từng khung hình.
              </p>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
              {recentVideos.length > 0 && (
                <Select
                  id="select-recent-video"
                  labelText="Chọn video đã xử lý"
                  hideLabel
                  value={selectedVideoId}
                  onChange={(e) => void loadVideoResult(e.target.value)}
                >
                  {recentVideos.map((v) => (
                    <SelectItem
                      key={v.id}
                      value={v.id}
                      text={`${v.original_name} (${v.event_count} sự kiện · ${v.detection_count} detections)`}
                    />
                  ))}
                </Select>
              )}
              <Button
                size="sm"
                kind={showOverlay ? "primary" : "secondary"}
                onClick={() => setShowOverlay((v) => !v)}
              >
                {showOverlay ? "Đang bật Khung Bounding Box AI" : "Bật Khung Bounding Box AI"}
              </Button>
            </div>
          </div>

          {loadingResult && (
            <div style={{ padding: 24 }}>
              <InlineLoading description="Đang tải luồng video đã xử lý và dữ liệu Bounding Box..." />
            </div>
          )}

          {!loadingResult && !analysisResult && (
            <div style={{ padding: 24 }}>
              <EmptyState
                title="Chưa có video nào được tải lên"
                copy="Hãy chọn một tệp video ở khung phía trên (ví dụ: data/smoke/dmd_smoke_clip.mp4) và bấm 'Bắt đầu Phân tích Video' để xem kết quả trực tiếp tại đây."
              />
            </div>
          )}

          {!loadingResult && analysisResult && (
            <div className="processed-workspace">
              {/* Thanh chỉ số tóm tắt của video vừa xử lý */}
              <div className="processed-kpi-bar">
                <div>
                  <span>Tên tệp Video</span>
                  <strong>{analysisResult.video.original_name}</strong>
                </div>
                <div>
                  <span>Trạng thái Xử lý</span>
                  <strong>
                    <Tag type={analysisResult.job?.status === "COMPLETED" ? "green" : "blue"}>
                      {analysisResult.job?.status ?? "COMPLETED"} (
                      {Math.round((analysisResult.job?.progress ?? 1) * 100)}%)
                    </Tag>
                  </strong>
                </div>
                <div>
                  <span>Sự kiện Vi phạm Phát hiện</span>
                  <strong>{analysisResult.summary.total_events} sự kiện</strong>
                </div>
                <div>
                  <span>Tổng số Khung nhận diện (Detections)</span>
                  <strong>
                    {analysisResult.summary.total_detections} boxes (
                    {analysisResult.detections.filter((d) => d.class_name.includes("phone")).length} phone ·{" "}
                    {analysisResult.detections.filter((d) => d.class_name === "seatbelt_fastened").length} seatbelt ·{" "}
                    {analysisResult.detections.filter((d) => d.class_name === "seatbelt_unfastened").length} no_seatbelt)
                  </strong>
                </div>
              </div>

              {/* Trình phát Video kèm Canvas Bounding Box */}
              <div className="processed-player-grid">
                <div className="video-stage-card">
                  <div className="video-stage-header">
                    <strong>
                      Trình phát Video Đã Phân tích + Lớp phủ Bounding Box (t = {currentVideoTime.toFixed(2)}s)
                    </strong>
                    <div className="legend-chips">
                      <span className="legend-chip act-call-right">Khung Đỏ: DRIVER PHONE (Vi phạm)</span>
                      <span className="legend-chip act-text-left">Khung Xanh Dương: PASSENGER PHONE (Không vi phạm)</span>
                      <span className="legend-chip act-call-left">Khung Cam: NO_SEATBELT (Vi phạm mọi ghế)</span>
                      <span className="legend-chip act-text-right">Khung Xanh Lá: SEATBELT (Đã thắt dây)</span>
                    </div>
                  </div>
                  {videoBlobUrl ? (
                    <div className="video-canvas-wrapper">
                      <video
                        ref={videoRef}
                        src={videoBlobUrl}
                        controls
                        preload="metadata"
                        onLoadedMetadata={drawOverlay}
                        onTimeUpdate={(e) => {
                          setCurrentVideoTime(e.currentTarget.currentTime);
                          drawOverlay();
                        }}
                        onSeeked={(e) => {
                          setCurrentVideoTime(e.currentTarget.currentTime);
                          drawOverlay();
                        }}
                      />
                      <canvas ref={canvasRef} className="bbox-overlay-canvas" />
                    </div>
                  ) : (
                    <div className="evidence-placeholder">
                      <Video size={40} />
                      <strong>Không tìm thấy tệp luồng video gốc</strong>
                    </div>
                  )}

                  {/* Các nút bấm tua nhanh tới từng sự kiện phát hiện */}
                  {analysisResult.events.length > 0 && (
                    <div className="quick-seek-bar">
                      <span>Tua nhanh tới sự kiện vi phạm:</span>
                      {analysisResult.events.map((ev, idx) => (
                        <Button
                          key={ev.id}
                          size="sm"
                          kind="tertiary"
                          renderIcon={Play}
                          onClick={() => seekVideoTo(ev.timestamp_seconds)}
                        >
                          Sự kiện #{idx + 1}: {ev.event_type} lúc {ev.timestamp_seconds.toFixed(2)}s (
                          {Math.round(ev.confidence * 100)}%)
                        </Button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Cột bên phải: Các Sự kiện & Video Bằng chứng Đã Xử lý (evidence.mp4 + annotated_keyframe.jpg) */}
                <div className="processed-events-column">
                  <h3 className="processed-subheading">
                    Sự kiện Vi phạm &amp; Video Bằng chứng Đã Xử lý ({analysisResult.events.length})
                  </h3>
                  {analysisResult.events.length === 0 ? (
                    <p style={{ color: "var(--muted)", fontSize: 13 }}>
                      Không phát hiện sự kiện vi phạm vượt ngưỡng trong video này.
                    </p>
                  ) : (
                    <div className="processed-events-list">
                      {analysisResult.events.map((ev) => (
                        <ProcessedEventCard key={ev.id} event={ev} onSeekVideo={seekVideoTo} />
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Bảng danh sách Detections theo từng khung hình (Frame-by-Frame Detections) */}
              {analysisResult.detections.length > 0 && (
                <div style={{ marginTop: 18 }}>
                  <div className="panel-heading" style={{ borderTop: "1px solid var(--rule)" }}>
                    <div>
                      <h3>Danh sách Nhận diện Theo Từng Khung hình (Frame-by-Frame Detections)</h3>
                      <p>
                        Bấm vào bất kỳ dòng nào để tua trình phát video tới đúng thời điểm và xem khung{" "}
                        <code>Bounding Box</code> tương ứng.
                      </p>
                    </div>
                    <Tag type="teal">{analysisResult.detections.length} Detections</Tag>
                  </div>
                  <div className="table-scroll" style={{ maxHeight: 280 }}>
                    <table className="event-table">
                      <thead>
                        <tr>
                          <th>Frame</th>
                          <th>Thời điểm (s)</th>
                          <th>Vai trò trong Vehicle</th>
                          <th>Lớp đối tượng (Class)</th>
                          <th>Kết luận Luật Giao thông</th>
                          <th>Confidence</th>
                          <th>Tư thế (Posture)</th>
                          <th>Hành vi 4 Lớp (Source Action)</th>
                          <th>Tọa độ Bounding Box [x1, y1, x2, y2]</th>
                          <th>Thao tác</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analysisResult.detections.map((d) => {
                          const role = (d.occupant_role || "driver").toLowerCase();
                          const isDriver = role === "driver";
                          const isPhone = d.class_name.includes("phone");
                          const isUnfastened = d.class_name.includes("unfastened");
                          const isViolation = (isPhone && isDriver) || isUnfastened;
                          const policyText = d.bbox?.violation_label_vi
                            ? String(d.bbox.violation_label_vi)
                            : isPhone
                              ? isDriver
                                ? "VI PHẠM (Tài xế dùng ĐT)"
                                : "KHÔNG VI PHẠM (Hành khách được phép dùng ĐT)"
                              : isUnfastened
                                ? isDriver
                                  ? "VI PHẠM (Tài xế không thắt dây đai)"
                                  : "VI PHẠM (Hành khách không thắt dây đai)"
                                : "KHÔNG VI PHẠM (Đã thắt dây đai an toàn)";
                          return (
                            <tr key={d.id}>
                              <td className="number">{d.frame_number}</td>
                              <td className="number">{d.timestamp_seconds.toFixed(2)}s</td>
                              <td>
                                <Tag type={isDriver ? "purple" : "cyan"}>
                                  {isDriver ? "Tài xế (driver)" : "Hành khách (front_passenger)"}
                                </Tag>
                              </td>
                              <td>
                                <Tag
                                  type={
                                    isPhone
                                      ? isDriver
                                        ? "red"
                                        : "blue"
                                      : isUnfastened
                                        ? "magenta"
                                        : d.class_name.includes("fastened")
                                          ? "green"
                                          : "blue"
                                  }
                                >
                                  {d.class_name === "seatbelt_fastened"
                                    ? "seatbelt (seatbelt_fastened)"
                                    : d.class_name === "seatbelt_unfastened"
                                      ? "no_seatbelt (seatbelt_unfastened)"
                                      : d.class_name}
                                </Tag>
                              </td>
                              <td>
                                <Tag
                                  type={
                                    isViolation
                                      ? "red"
                                      : isPhone && !isDriver
                                        ? "blue"
                                        : "green"
                                  }
                                >
                                  {policyText}
                                </Tag>
                              </td>
                              <td className="number">{(d.confidence * 100).toFixed(1)}%</td>
                              <td className="mono">{String(d.bbox?.phone_posture ?? "—")}</td>
                              <td className="mono">
                                {d.bbox?.source_action
                                  ? String(d.bbox.source_action).replace("driver_actions/", "")
                                  : "—"}
                              </td>
                              <td className="mono">
                                {Array.isArray(d.bbox?.xyxy)
                                  ? `[${d.bbox.xyxy.map((v) => Math.round(Number(v))).join(", ")}]`
                                  : "—"}
                              </td>
                              <td>
                                <Button
                                  size="sm"
                                  kind="ghost"
                                  onClick={() => seekVideoTo(d.timestamp_seconds)}
                                >
                                  Xem frame này
                                </Button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </Shell>
  );
}

function EventsPage() {
  const [events, setEvents] = useState<SafetyEvent[]>([]); const [loading, setLoading] = useState(true); const [error, setError] = useState(""); const [type, setType] = useState(""); const [status, setStatus] = useState(""); const [exporting, setExporting] = useState(false);
  useEffect(() => { const query = new URLSearchParams(); if (type) query.set("event_type", type); if (status) query.set("review_status", status); api.events(`?${query}`).then(setEvents).catch((reason) => setError(reason.message)).finally(() => setLoading(false)); }, [type, status]);
  async function exportCsv() {
    setExporting(true); setError("");
    try {
      const { blob, filename } = await api.exportEvents();
      const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
      anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Xuất CSV thất bại"); }
    finally { setExporting(false); }
  }
  return <Shell><main><PageHeader title="Nhật ký Sự kiện Vi phạm (Event Ledger)" description="Tra cứu các sự kiện vi phạm, kiểm tra ảnh/video bằng chứng và lưu quyết định phê duyệt." action={<Button kind="tertiary" renderIcon={DocumentExport} disabled={exporting} onClick={exportCsv}>{exporting ? "Đang xuất..." : "Xuất báo cáo CSV"}</Button>} />
    <section className="filter-sheet"><Select id="event-type" labelText="Loại sự kiện (Event Type)" value={type} onChange={(event) => setType(event.target.value)}><SelectItem value="" text="Tất cả loại vi phạm" /><SelectItem value="PHONE" text="PHONE (Sử dụng điện thoại)" /><SelectItem value="NO_SEATBELT" text="NO_SEATBELT (Không thắt dây an toàn)" /></Select><Select id="review-status" labelText="Trạng thái kiểm duyệt (Review State)" value={status} onChange={(event) => setStatus(event.target.value)}><SelectItem value="" text="Tất cả trạng thái" /><SelectItem value="PENDING" text="Chờ duyệt (PENDING)" /><SelectItem value="NEEDS_REVIEW" text="Cần kiểm tra (NEEDS_REVIEW)" /><SelectItem value="CONFIRMED" text="Đã xác nhận (CONFIRMED)" /><SelectItem value="REJECTED" text="Từ chối (REJECTED)" /></Select></section>
    <section className="record-panel full-table"><div className="panel-heading"><div><h2>Danh sách Sự kiện Ghi nhận</h2><p>{events.length} bản ghi phù hợp</p></div></div>{error && <InlineNotification kind="error" title="Không thể tải danh sách sự kiện" subtitle={error} lowContrast hideCloseButton />}{loading ? <TableSkeleton /> : <EventTable events={events} />}</section>
  </main></Shell>;
}

const TRACE_DISPLAY_KEYS = [
  "subject_id",
  "vehicle_track_id",
  "cabin_bbox",
  "cabin_method",
  "cabin_confidence",
  "behavior_track_id",
  "occupant_association_method",
  "occupant_role_confidence",
  "phone_context",
  "phone_posture",
  "source_action",
  "pose_confidence",
  "fusion_mode",
  "temporal_score",
  "iou_with_gt",
  "start_latency_seconds",
  "end_latency_seconds",
  "ground_truth_window",
  "predicted_window",
  "evidence_source",
  "evidence_status",
];

function EventDetailPage() {
  const { id = "" } = useParams();
  const [event, setEvent] = useState<EventDetail | null>(null);
  const [loadError, setLoadError] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const { assets, loading: evidenceLoading, error: evidenceError } = useEvidenceAssets(event);
  useEffect(() => { setLoadError(""); api.event(id).then(setEvent).catch((reason) => setLoadError(reason instanceof Error ? reason.message : "Không thể tải thông tin sự kiện")); }, [id]);
  async function review(status: ReviewStatus) {
    if (!event || saving) return;
    setSaving(true); setReviewError("");
    try { await api.review(event.id, status, notes); setEvent(await api.event(event.id)); setNotes(""); }
    catch (reason) { setReviewError(reason instanceof Error ? reason.message : "Không thể lưu quyết định kiểm duyệt"); }
    finally { setSaving(false); }
  }
  return <Shell><main><PageHeader title="Kiểm tra Chi tiết Sự kiện & Bằng chứng" description="Đối chiếu hình ảnh keyframe, video bằng chứng, nhật ký nhân quả V32 (Causal Trace) và phê duyệt sự kiện." />
    {loadError && <><InlineNotification kind="error" title="Không tìm thấy sự kiện" subtitle={`${loadError}. Vui lòng quay lại Nhật ký Sự kiện.`} lowContrast hideCloseButton /><Button className="recovery-action" kind="tertiary" as={Link} to="/events">Quay lại Nhật ký Sự kiện</Button></>}
    {!event && !loadError ? <InlineLoading description="Đang tải hồ sơ sự kiện..." /> : event ? <>
      {event.review_status === "NEEDS_REVIEW" && <InlineNotification className="readiness-alert" kind="warning" title="Bằng chứng cần kiểm tra thủ công" subtitle="Hệ thống phát hiện vùng che khuất hoặc độ không chắc chắn cao. Vui lòng xem kỹ hình ảnh trước khi xác nhận." lowContrast hideCloseButton />}
      <div className="inspection-layout"><section className="evidence-sheet" aria-labelledby="evidence-heading"><div className="panel-heading"><div><h2 id="evidence-heading">Hồ sơ Bằng chứng Ghi nhận</h2><p>Ảnh keyframe, video clip bằng chứng và dữ liệu kiểm toán nhân quả V32</p></div></div>
        {evidenceLoading && <div className="evidence-loading"><InlineLoading description="Đang tải tệp bằng chứng..." /></div>}
        {evidenceError && <InlineNotification kind="error" title="Không thể tải tệp bằng chứng" subtitle={evidenceError} lowContrast hideCloseButton />}
        {!evidenceLoading && !evidenceError && !event.evidence.available && <div className="evidence-placeholder"><Video size={48} /><strong>{event.evidence_trace?.evidence_status === "V32_BENCHMARK_TRACE" ? "Sự kiện Thực nghiệm V32 Benchmark Trace" : "Tệp bằng chứng chưa đầy đủ"}</strong><span>{event.evidence_trace?.evidence_status === "V32_BENCHMARK_TRACE" ? `Sự kiện nhân quả đã kiểm chứng từ đối tượng ${String(event.evidence_trace.subject_id || event.video_id)} (Khoảng dự đoán: ${String(event.evidence_trace.predicted_window || "")}). Xem chi tiết thông số kỹ thuật bên dưới hoặc mở trang Đánh giá Thực nghiệm V32.` : `${event.evidence.integrity_errors.join(" ") || "Sự kiện này chưa có tệp video bằng chứng."}`}</span></div>}
        {!evidenceLoading && assets.original && <div className="evidence-frames"><figure><img src={assets.original} alt="Ảnh gốc tại thời điểm phát hiện" /><figcaption>Ảnh Keyframe Gốc (Original Keyframe)</figcaption></figure>{assets.annotated && <figure><img src={assets.annotated} alt="Ảnh đã vẽ khung Bounding Box nhận diện" /><figcaption>Ảnh Keyframe Đánh dấu (Annotated Keyframe)</figcaption></figure>}</div>}
        {!evidenceLoading && assets.clip && <div className="evidence-clip"><h3>Video Bằng chứng Theo Thời gian (Temporal Evidence Clip)</h3><video controls preload="metadata" src={assets.clip}>Trình duyệt không hỗ trợ phát video này.</video></div>}
        <div className="frame-meta"><span>Frame {event.frame_number}</span><span>{formatTimestamp(event.timestamp_seconds)}</span><span>Track {event.track_id ?? "Chưa gán"}</span></div></section>
        <aside className="decision-sheet"><h2>Phiếu Quyết định Kiểm duyệt</h2><dl><div><dt>Loại sự kiện</dt><dd>{event.event_type === "PHONE" ? "PHONE (Điện thoại)" : "NO_SEATBELT (Dây an toàn)"}</dd></div><div><dt>Vị trí người ngồi</dt><dd>{event.occupant_role.replaceAll("_", " ")}</dd></div><div><dt>Loại phương tiện</dt><dd>{event.vehicle_type.replaceAll("_", " ")}</dd></div><div><dt>Mã ngữ cảnh xe</dt><dd>{event.vehicle_context_id}</dd></div><div><dt>Confidence Score</dt><dd>{Math.round(event.confidence * 100)}%</dd></div><div><dt>Fusion Score</dt><dd>{event.fusion_score == null ? "NOT_AVAILABLE" : `${Math.round(event.fusion_score * 100)}%`}</dd></div><div><dt>Phiên bản Model</dt><dd>{event.model_version}</dd></div><div><dt>Trạng thái hiện tại</dt><dd><StatusTag status={event.review_status} /></dd></div></dl>
          {reviewError && <InlineNotification kind="error" title="Chưa lưu được quyết định" subtitle={reviewError} lowContrast hideCloseButton />}
          <TextArea id="review-notes" labelText="Ghi chú kiểm duyệt" helperText={`${notes.length}/2000 ký tự. Ghi rõ bằng chứng quan sát được.`} maxLength={2000} value={notes} onChange={(event) => setNotes(event.target.value)} />
          <div className="decision-actions"><Button size="sm" disabled={saving || !event.evidence.confirmation_ready} onClick={() => review("CONFIRMED")}>{saving ? "Đang lưu..." : "Xác nhận (Confirm)"}</Button><Button size="sm" kind="secondary" disabled={saving} onClick={() => review("NEEDS_REVIEW")}>Cần xem lại</Button><Button size="sm" kind="danger--tertiary" disabled={saving} onClick={() => review("REJECTED")}>Từ chối (Reject)</Button></div>
          {!event.evidence.confirmation_ready && <p className="decision-guard">Nút Xác nhận (Confirm) chỉ bật khi gói bằng chứng video có đầy đủ mã băm toàn vẹn SHA-256.</p>}
        </aside></div>
      <section className="audit-grid"><div className="record-panel trace-panel"><div className="panel-heading"><div><h2>Nhật ký Truy vết Quyết định (Decision Trace)</h2><p>Thông số ngữ cảnh, tư thế <code>phone_posture</code>, hành vi 4 lớp <code>source_action</code> và điểm số thời gian</p></div></div><dl>{Object.entries(event.evidence_trace).filter(([key]) => TRACE_DISPLAY_KEYS.includes(key)).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{typeof value === "number" ? value.toFixed(3) : Array.isArray(value) ? JSON.stringify(value) : String(value)}</dd></div>)}</dl></div>
        <div className="record-panel history-panel"><div className="panel-heading"><div><h2>Lịch sử Kiểm duyệt (Review History)</h2><p>Nhật ký bất biến ghi nhận các lần phê duyệt</p></div></div>{event.review_history.length ? <ol>{event.review_history.map((record, index) => <li key={`${record.created_at}-${index}`}><div><strong>{record.previous_status} → {record.new_status}</strong><time dateTime={record.created_at}>{new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(record.created_at))}</time></div><p>{record.notes || "Không có ghi chú."}</p><small>Người duyệt: {record.reviewer_id}</small></li>)}</ol> : <div className="history-empty">Chưa có quyết định kiểm duyệt thủ công nào.</div>}</div></section>
    </> : null}</main></Shell>;
}

function ResearchReviewPage() {
  const [queue, setQueue] = useState<PhoneReviewQueue | null>(null);
  const [selectedCase, setSelectedCase] = useState("");
  const [detail, setDetail] = useState<PhoneReviewDetail | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [visibility, setVisibility] = useState<PhoneVisibility>("UNRESOLVED");
  const [phoneUse, setPhoneUse] = useState<PhoneUseAssessment>("UNRESOLVED");
  const [proposalId, setProposalId] = useState("NO_VALID_PROPOSAL");
  const [notes, setNotes] = useState("");
  const [preReviewRevision, setPreReviewRevision] = useState<string | null>(null);

  function loadQueue(preferredCase?: string) {
    setLoading(true);
    return api.phoneReviews()
      .then((result) => {
        setQueue(result);
        const available = new Set(result.cases.map((item) => item.case_id));
        const next = preferredCase && available.has(preferredCase)
          ? preferredCase
          : result.cases.find((item) => item.review_status !== "REVIEWED")?.case_id
            ?? result.cases[0]?.case_id
            ?? "";
        setSelectedCase(next);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Không thể tải hàng đợi kiểm duyệt nghiên cứu"))
      .finally(() => setLoading(false));
  }

  useEffect(() => { void loadQueue(); }, []);

  useEffect(() => {
    if (!selectedCase) { setDetail(null); setImageUrl(""); return; }
    const controller = new AbortController();
    let objectUrl = "";
    setError("");
    Promise.all([api.phoneReview(selectedCase), api.phoneReviewImage(selectedCase, controller.signal)])
      .then(([record, blob]) => {
        objectUrl = URL.createObjectURL(blob);
        setDetail(record);
        setImageUrl(objectUrl);
        const startingPoint = record.decision ?? record.pre_review;
        setVisibility(startingPoint?.phone_visibility ?? "UNRESOLVED");
        setPhoneUse(startingPoint?.phone_use_assessment ?? "UNRESOLVED");
        setProposalId(startingPoint?.selected_proposal_id ?? "NO_VALID_PROPOSAL");
        setNotes(record.decision?.notes ?? record.pre_review?.rationale ?? "");
        setPreReviewRevision(
          !record.decision || record.pre_review_changed_since_decision
            ? record.pre_review?.revision ?? null
            : null,
        );
      })
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Không thể tải ảnh contact-sheet");
      });
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [selectedCase]);

  function changeVisibility(value: PhoneVisibility) {
    setVisibility(value);
    if (value === "NO_PHONE_VISIBLE") {
      setPhoneUse("NOT_PHONE_USE");
      setProposalId("NO_VALID_PROPOSAL");
    } else if (value === "UNRESOLVED") {
      setPhoneUse("UNRESOLVED");
      setProposalId("NO_VALID_PROPOSAL");
    }
  }

  function changePhoneUse(value: PhoneUseAssessment) {
    setPhoneUse(value);
    if (value === "PHONE_USE_CONFIRMED") setVisibility("PHONE_VISIBLE");
  }

  function applyPreReview() {
    if (!detail?.pre_review) return;
    setVisibility(detail.pre_review.phone_visibility);
    setPhoneUse(detail.pre_review.phone_use_assessment);
    setProposalId(detail.pre_review.selected_proposal_id);
    setNotes(detail.pre_review.rationale);
    setPreReviewRevision(detail.pre_review.revision);
  }

  async function saveReview() {
    if (!detail) return;
    setSaving(true);
    setError("");
    setMessage("");
    const payload: PhoneReviewInput = {
      phone_visibility: visibility,
      phone_use_assessment: phoneUse,
      selected_proposal_id: proposalId,
      notes: notes.trim(),
      expected_revision: detail.decision?.revision ?? null,
      pre_review_revision: preReviewRevision,
    };
    try {
      await api.savePhoneReview(detail.case_id, payload);
      const refreshed = await api.phoneReviews();
      setQueue(refreshed);
      const next = refreshed.cases.find((item) => item.review_status !== "REVIEWED")?.case_id;
      setMessage(`Đã lưu quyết định kiểm duyệt cho ${detail.case_id}.`);
      setSelectedCase(next ?? detail.case_id);
      if (!next) {
        const updated = await api.phoneReview(detail.case_id);
        setDetail(updated);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Không thể lưu quyết định");
    } finally {
      setSaving(false);
    }
  }

  const reviewed = queue?.summary.reviewed ?? 0;
  const total = queue?.summary.total ?? 13;
  return <Shell><main>
    <PageHeader title="Kiểm duyệt Nghiên cứu Đề xuất Điện thoại (Phone Proposal Review)" description="Xác minh thủ công các trường hợp khó (False Negative / Proposal) trong quá trình nghiên cứu." />
    <section className="review-progress" aria-label="Tiến độ kiểm duyệt">
      <div><span>Đã duyệt (Reviewed)</span><strong>{reviewed}</strong></div>
      <div><span>Chờ duyệt (Pending)</span><strong>{queue?.summary.pending ?? total}</strong></div>
      <div><span>Cần kiểm tra lại</span><strong>{queue?.summary.recheck ?? 0}</strong></div>
      <div><span>Tỷ lệ hoàn thành</span><strong>{Math.round((reviewed / Math.max(1, total)) * 100)}%</strong></div>
    </section>
    {error && <InlineNotification className="readiness-alert" kind="error" title="Không thể tải dữ liệu kiểm duyệt" subtitle={error} lowContrast hideCloseButton />}
    {message && <InlineNotification className="readiness-alert" kind="success" title="Đã ghi nhận kết quả" subtitle={message} lowContrast hideCloseButton />}
    {loading ? <InlineLoading description="Đang tải danh sách tình huống nghiên cứu..." /> : <div className="research-review-layout">
      <aside className="review-case-list" aria-label="Danh sách tình huống">
        <div className="panel-heading"><div><h2>13 Tình huống Nghiên cứu</h2><p>Dữ liệu kiểm chứng phát triển</p></div></div>
        <div className="case-list-scroll">{queue?.cases.map((item) => <button type="button" key={item.case_id} className={selectedCase === item.case_id ? "active" : ""} onClick={() => setSelectedCase(item.case_id)} aria-current={selectedCase === item.case_id ? "true" : undefined}>
          <span><strong>{item.case_id}</strong><small>{item.subject_id} · {item.sample_count} frames</small></span>
          <Tag type={item.review_status === "REVIEWED" ? "green" : item.review_status === "RECHECK" ? "red" : "warm-gray"}>{item.review_status}</Tag>
        </button>)}</div>
      </aside>
      <section className="research-review-workspace">
        {!detail ? <div className="empty-state"><Review size={32} /><h3>Chọn một tình huống</h3><p>Chọn một mã case bên trái để xem ảnh contact-sheet.</p></div> : <>
          <div className="review-evidence-head">
            <div><p className="eyebrow">Bằng chứng Tình huống</p><h2>{detail.case_id}</h2><p>{detail.sample_count} khung hình mẫu · {detail.specialist_proposals} specialist · {detail.auxiliary_proposals} COCO proposals</p></div>
            <div className="review-flags"><Tag type={detail.cross_model_agreement ? "green" : "gray"}>{detail.cross_model_agreement ? "Cross-model agreement" : "No cross-model agreement"}</Tag><Tag type="warm-gray">Research Audit</Tag></div>
          </div>
          <div className="contact-sheet-frame">
            {imageUrl ? <a href={imageUrl} target="_blank" rel="noreferrer" title="Mở ảnh độ phân giải gốc"><img src={imageUrl} alt={`Contact sheet của ${detail.case_id}`} /></a> : <InlineLoading description="Đang tải ảnh contact sheet..." />}
            <p>Cuộn ngang để xem từng khung hình. Bấm vào ảnh để phóng to toàn màn hình.</p>
          </div>
          <div className="review-record-grid">
            <div className="decision-form">
              <h3>Quyết định của Người kiểm duyệt</h3>
              {detail.pre_review && (!detail.decision || detail.pre_review_changed_since_decision) && <div className="pre-review-card">
                <div><Tag type="blue">{detail.decision ? "AI RE-REVIEW" : "AI PRE-REVIEW"}</Tag><strong>{detail.pre_review.confidence} confidence</strong></div>
                <div className="pre-review-fields"><span>{detail.pre_review.phone_visibility.replaceAll("_", " ")}</span><span>{detail.pre_review.phone_use_assessment.replaceAll("_", " ")}</span><span>{detail.pre_review.selected_proposal_id}</span></div>
                <p>{detail.pre_review.rationale}</p>
                {detail.decision && <Button kind="tertiary" size="sm" onClick={applyPreReview}>Áp dụng gợi ý AI</Button>}
              </div>}
              <Select id="phone-visibility" labelText="Có nhìn thấy điện thoại không? (Phone Visibility)" value={visibility} onChange={(event) => changeVisibility(event.target.value as PhoneVisibility)}>
                <SelectItem value="UNRESOLVED" text="Chưa rõ (UNRESOLVED)" />
                <SelectItem value="PHONE_VISIBLE" text="Có nhìn thấy điện thoại (PHONE_VISIBLE)" />
                <SelectItem value="NO_PHONE_VISIBLE" text="Không thấy điện thoại (NO_PHONE_VISIBLE)" />
              </Select>
              <Select id="phone-use" labelText="Tài xế có đang sử dụng điện thoại không?" value={phoneUse} onChange={(event) => changePhoneUse(event.target.value as PhoneUseAssessment)}>
                <SelectItem value="UNRESOLVED" text="Chưa rõ (UNRESOLVED)" />
                <SelectItem value="PHONE_USE_CONFIRMED" text="Xác nhận đang sử dụng (PHONE_USE_CONFIRMED)" />
                <SelectItem value="NOT_PHONE_USE" text="Không sử dụng (NOT_PHONE_USE)" />
              </Select>
              <Select id="accepted-proposal" labelText="Khung đề xuất hợp lệ tốt nhất (Best Proposal)" value={proposalId} onChange={(event) => setProposalId(event.target.value)} disabled={visibility !== "PHONE_VISIBLE"}>
                <SelectItem value="NO_VALID_PROPOSAL" text="Không có proposal hợp lệ (NO_VALID_PROPOSAL)" />
                {detail.proposals.map((proposal) => <SelectItem key={proposal.proposal_id} value={proposal.proposal_id} text={`${proposal.proposal_id} · ${proposal.source === "specialist" ? "Specialist" : "COCO"} · ${(proposal.confidence * 100).toFixed(1)}% · ${proposal.geometry_context.replaceAll("_", " ")}`} />)}
              </Select>
              <TextArea id="review-notes" labelText="Lý do & Nhận xét chi tiết" helperText="Bắt buộc (tối thiểu 12 ký tự)." rows={4} value={notes} onChange={(event) => setNotes(event.target.value)} />
              <Button renderIcon={Save} onClick={saveReview} disabled={saving || notes.trim().length < 12}>{saving ? "Đang lưu..." : "Lưu Quyết định Kiểm duyệt"}</Button>
            </div>
            <aside className="proposal-register">
              <h3>Danh sách Khung đề xuất (Proposals)</h3>
              <p>Chọn mã ID khớp với khung điện thoại trên ảnh contact sheet.</p>
              <div>{detail.proposals.map((proposal) => <button type="button" key={proposal.proposal_id} className={proposalId === proposal.proposal_id ? "selected" : ""} onClick={() => { setVisibility("PHONE_VISIBLE"); setProposalId(proposal.proposal_id); }}>
                <strong>{proposal.proposal_id}</strong><span>{proposal.source === "specialist" ? "Specialist" : "COCO"}</span><span>{(proposal.confidence * 100).toFixed(1)}%</span><small>Frame {proposal.frame_index} · hand {proposal.hand_proximity.toFixed(2)} · face {proposal.face_proximity.toFixed(2)}</small>
              </button>)}</div>
            </aside>
          </div>
          {detail.decision && <div className="prior-decision"><strong>Biên nhận gần nhất</strong><span>{new Date(detail.decision.reviewed_at_utc).toLocaleString()}</span><span className="mono">{detail.decision.revision.slice(0, 12)}</span></div>}
        </>}
      </section>
    </div>}
  </main></Shell>;
}

function BenchmarkShellPage() {
  return (
    <Shell>
      <main>
        <PageHeader
          title="Đánh giá Thực nghiệm Khoa học V32 AI85 & Kiểm toán Nhân quả"
          description="Kết quả đánh giá chính thức trên tập dữ liệu DMD (gC-14, gZ-36), hiệu chuẩn dây an toàn Seatbelt S3 và kiểm chứng phần cứng GPU Kaggle Tesla T4."
        />
        <BenchmarkPage />
      </main>
    </Shell>
  );
}

function StatisticsPage() {
  const { stats, loading, error } = useDashboardData();
  const v32 = stats.v32_summary ?? emptyStats.v32_summary!;
  return <Shell><main><PageHeader title="Thống kê Vận hành & Chỉ số Khoa học V32" description="Số liệu sự kiện trong cơ sở dữ liệu kết hợp cùng bảng chỉ số thực nghiệm V32 AI85 đã được kiểm chứng." action={<Button kind="secondary" as={Link} to="/benchmark" renderIcon={DataCheck}>Mở Bảng Thực nghiệm V32 Chi tiết</Button>} />{error && <InlineNotification kind="error" title="Không thể tải thống kê" subtitle={error} lowContrast hideCloseButton />}{loading ? <InlineLoading description="Đang tải số liệu thống kê..." /> : !error && stats ? <><MetricStrip stats={stats} /><div className="statistics-grid"><section className="record-panel"><h2>Phân bố Sự kiện Trong Cơ sở Dữ liệu</h2><Distribution stats={stats} /></section><section className="record-panel"><h2>Chỉ số Thực nghiệm Khoa học V32 AI85</h2><dl className="metric-status"><div><dt>Track A (Frozen Temporal)</dt><dd><strong>{(v32.track_a_f1 * 100).toFixed(2)}% F1</strong> (Precision: {(v32.track_a_precision * 100).toFixed(1)}%, Recall: {(v32.track_a_recall * 100).toFixed(1)}%)</dd></div><div><dt>Track B (4-Class Exact Action)</dt><dd><strong>{(v32.track_b_f1 * 100).toFixed(2)}% F1</strong> (19/19 TP)</dd></div><div><dt>Track C (Generic Episode Gap 3s)</dt><dd><strong>{(v32.track_c_f1 * 100).toFixed(2)}% F1</strong> (16/16 TP)</dd></div><div><dt>Seatbelt V32 Binary (n=616)</dt><dd><strong>{(v32.seatbelt_binary_accuracy * 100).toFixed(2)}% Accuracy</strong> ({(v32.seatbelt_binary_macro_f1 * 100).toFixed(2)}% Macro-F1)</dd></div><div><dt>Seatbelt V32 3-Class (n=641)</dt><dd><strong>{(v32.seatbelt_3class_balanced_accuracy * 100).toFixed(2)}% Balanced Acc</strong></dd></div><div><dt>Trạng thái Cổng AI85</dt><dd><Tag type="green">{stats.event_metrics_status}</Tag></dd></div></dl><p className="scientific-note">Toàn bộ các chỉ số V32 được xác thực bằng mã băm SHA-256 trong thư mục <code>reports/v32_ai85_execution/</code>.</p></section></div></> : null}</main></Shell>;
}

function ModelsPage() {
  return <Shell><main><PageHeader title="Mô hình AI & Kiến trúc Phương pháp luận" description="Sự tiến hóa từ mô hình đơn khối V1 Baseline sang kiến trúc đa tầng V2 và hệ thống suy luận nhân quả V32 AI85." action={<Button as={Link} to="/benchmark" renderIcon={DataCheck}>Xem Thực nghiệm V32</Button>} />
    <div style={{ display: "grid", gap: 16, marginTop: 18 }}>
      <section className="model-sheet" style={{ marginTop: 0, maxWidth: "none" }}>
        <div>
          <div className="v32-badges"><Tag type="green">KIẾN TRÚC ĐỒ ÁN HIỆN TẠI</Tag><Tag type="blue">97.44% F1 TRÊN TRACK A &amp; B</Tag></div>
          <h2>V32 AI85 — Kiến trúc Đa tầng Nhân quả Giám sát An toàn Tài xế</h2>
          <p>Kết hợp nhận diện điện thoại đa vùng cắt (D5 Specialist), suy luận khoảng cách giải phẫu chuẩn hóa theo khung xương <code>YOLO11-pose</code>, phân loại dây an toàn <code>Seatbelt S3</code> và bộ tách chuyển tiếp nhân quả.</p>
        </div>
        <dl>
          <div><dt>Phone Specialist (D5)</dt><dd>Quét toàn khung hình + vùng cắt <code>Driver ROI</code> (<code>models/active/v2/phone_detector.pt</code>) + đồng thuận COCO cell phone</dd></div>
          <div><dt>Anatomical Pose Reasoning</dt><dd><code>models/auxiliary/yolo11n-pose.pt</code> chuẩn hóa khoảng cách cổ tay &amp; tai/mặt theo đường chéo cơ thể (<code>norm_hand_prox</code>, <code>norm_face_prox</code>)</dd></div>
          <div><dt>Bộ lọc Nhiễu &amp; Chống Bắc cầu</dt><dd>Loại bỏ các dải viền dọc có tỷ lệ khung hình &gt; 3.0 và chặn hiện tượng tự bắc cầu (self-bridging) ở các khung đơn lẻ (&lt; 0.48)</dd></div>
          <div><dt>Causal Boundary Collector</dt><dd>Tách trực tiếp các pha chuyển tiếp liên tiếp <code>LAP_TEXTING</code> &harr; <code>EAR_CALL</code> với độ trễ kích hoạt trung bình 0.504s</dd></div>
          <div><dt>Seatbelt Classifier (V32 S3)</dt><dd><code>models/active/v2/seatbelt_classifier.pt</code> hiệu chuẩn ngưỡng không chắc chắn (&tau;<sub>u</sub>=0.055, margin=0.22) &rarr; 86.69% Binary Accuracy</dd></div>
          <div><dt>Trạng thái Kiểm định</dt><dd><Tag type="green">PASS_AI85_DEVELOPMENT_GATE_EXCEEDED</Tag></dd></div>
        </dl>
      </section>

      <section className="model-sheet" style={{ marginTop: 0, maxWidth: "none" }}>
        <div>
          <h2>MC_BOOTSTRAP_001 &amp; MULTIMODEL_V2 — Các Phiên bản Nền tảng</h2>
          <p>Được giữ nguyên nhằm đối sánh định lượng mức độ cải thiện qua các thế hệ V1 &rarr; V2 &rarr; V31 &rarr; V32.</p>
        </div>
        <dl>
          <div><dt>V1 Baseline</dt><dd>Một mô hình <code>YOLO11s</code> 3 lớp đơn lẻ (<code>phone</code>, <code>seatbelt_fastened</code>, <code>seatbelt_unfastened</code>)</dd></div>
          <div><dt>V2 Pipeline</dt><dd>Vehicle &rarr; Cabin &rarr; Occupant &rarr; Specialist detectors &rarr; Temporal hysteresis engine</dd></div>
          <div><dt>V31 Pre-Recovery Baseline</dt><dd>75.68% F1 (82.35% Precision, 70.00% Recall, 14 TP / 3 FP / 6 FN)</dd></div>
          <div><dt>Mức tăng trưởng V32 AI85</dt><dd><strong>+21.76% F1</strong> (+17.65% Precision lên 100.0%, +25.00% Recall lên 95.0%, 19 TP / 0 FP / 1 FN)</dd></div>
        </dl>
      </section>
    </div>
  </main></Shell>;
}

function SettingsPage() {
  const [driverSide, setDriverSide] = useState(() => localStorage.getItem("rw_driver_side") ?? "left");
  const [windowSec, setWindowSec] = useState(() => localStorage.getItem("rw_window_sec") ?? "2.0");
  const [minPosSec, setMinPosSec] = useState(() => localStorage.getItem("rw_min_pos_sec") ?? "0.50");
  const [maxAspect, setMaxAspect] = useState(() => localStorage.getItem("rw_max_aspect") ?? "3.0");
  const [seatbeltTau, setSeatbeltTau] = useState(() => localStorage.getItem("rw_seatbelt_tau") ?? "0.055");
  const [saved, setSaved] = useState(false);

  function saveSettings(e: FormEvent) {
    e.preventDefault();
    localStorage.setItem("rw_driver_side", driverSide);
    localStorage.setItem("rw_window_sec", windowSec);
    localStorage.setItem("rw_min_pos_sec", minPosSec);
    localStorage.setItem("rw_max_aspect", maxAspect);
    localStorage.setItem("rw_seatbelt_tau", seatbeltTau);
    setSaved(true);
    setTimeout(() => setSaved(false), 3500);
  }

  return <Shell><main><PageHeader title="Cấu hình Hệ thống & Tham số V32" description="Tùy chỉnh hình học khoang lái, bộ lọc giải phẫu V32 và ngưỡng thời gian nhân quả." />
    <form onSubmit={saveSettings} className="settings-sheet">
      <h2>Tham số Triển khai &amp; Hiệu chuẩn V32 AI85</h2>
      <Select id="driver-side" labelText="Vị trí ghế lái (Tọa độ khung hình)" value={driverSide} onChange={(e) => setDriverSide(e.target.value)}>
        <SelectItem value="left" text="Tay lái thuận / Bên trái ảnh (Mặc định chuẩn DMD)" />
        <SelectItem value="right" text="Tay lái nghịch / Bên phải ảnh" />
      </Select>
      <TextInput id="window-seconds" labelText="Cửa sổ thời gian xác nhận (Confirmation Window - giây)" value={windowSec} onChange={(e) => setWindowSec(e.target.value)} helperText="Cửa sổ trượt tích lũy ứng viên (Mặc định V32: 2.0s)." />
      <TextInput id="positive-seconds" labelText="Thời lượng dương tính tối thiểu (Minimum Positive Duration - giây)" value={minPosSec} onChange={(e) => setMinPosSec(e.target.value)} helperText="Yêu cầu duy trì liên tục để kích hoạt sự kiện nhân quả (Mặc định V32: 0.50s)." />
      <TextInput id="max-aspect-ratio" labelText="Tỷ lệ khung hình tối đa của điện thoại (Max Phone Aspect Ratio)" value={maxAspect} onChange={(e) => setMaxAspect(e.target.value)} helperText="Bộ lọc V32 Bug 3: loại bỏ các dải nhiễu dọc ở viền ghế/cột A có tỷ lệ > 3.0." />
      <TextInput id="seatbelt-tau" labelText="Ngưỡng độ không chắc chắn dây an toàn Seatbelt S3 (tau_u)" value={seatbeltTau} onChange={(e) => setSeatbeltTau(e.target.value)} helperText="Ngưỡng định tuyến lớp uncertain_or_occluded (Mặc định V32 S3: 0.055)." />
      {saved && <InlineNotification kind="success" title="Đã lưu cấu hình" subtitle="Đã cập nhật thông số vận hành V32 trên trình duyệt." lowContrast hideCloseButton />}
      <Button type="submit" renderIcon={Save}>Lưu cấu hình hệ thống</Button>
    </form>
  </main></Shell>;
}

function ProtectedApp() {
  if (!sessionStorage.getItem("roadwatch-token")) return <Navigate to="/login" replace />;
  return <Routes><Route path="/" element={<DashboardPage />} /><Route path="/benchmark" element={<BenchmarkShellPage />} /><Route path="/upload" element={<UploadPage />} /><Route path="/events" element={<EventsPage />} /><Route path="/events/:id" element={<EventDetailPage />} /><Route path="/research-review" element={<ResearchReviewPage />} /><Route path="/statistics" element={<StatisticsPage />} /><Route path="/models" element={<ModelsPage />} /><Route path="/settings" element={<SettingsPage />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes>;
}

export default function App() { return <Routes><Route path="/login" element={<LoginPage />} /><Route path="/*" element={<ProtectedApp />} /></Routes>; }
