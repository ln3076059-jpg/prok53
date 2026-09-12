'use strict';
const $ = id => document.getElementById(id);
const names = {
  rights: 'Quyền sử dụng', physical_lineage: 'Nguồn gốc vật lý', identity: 'Định danh đề xuất', sequence: 'Nội dung theo thời gian',
  PHONE_USE: 'Đang dùng điện thoại', PHONE_PRESENT_NOT_USED: 'Có điện thoại, chưa sử dụng',
  MOUNTED_OR_STATIC_PHONE: 'Điện thoại gắn cố định', NO_PHONE: 'Không có điện thoại', UNKNOWN: 'Chưa xác định',
  FASTENED: 'Đã cài dây an toàn', UNFASTENED: 'Chưa cài dây an toàn', UNCERTAIN_OR_OCCLUDED: 'Chưa rõ hoặc bị che khuất', NOT_APPLICABLE: 'Không áp dụng',
  driver: 'Người lái', front_passenger: 'Hành khách trước', rear_left: 'Hành khách sau trái', rear_center: 'Hành khách sau giữa', rear_right: 'Hành khách sau phải', unknown: 'Chưa rõ',
  clear: 'Rõ', partial: 'Một phần', occluded: 'Bị che khuất', out_of_view: 'Ngoài khung hình',
  PENDING: 'Chờ xác nhận', NOT_PROVABLE: 'Chưa chứng minh được', NEEDS_HUMAN_DECISION: 'Cần người có trách nhiệm quyết định',
  ACCEPT: 'Chấp nhận sử dụng', REJECT: 'Từ chối sử dụng', APPROVE: 'Xác nhận nguyên trạng', SUBMIT_FOR_VERIFICATION: 'Đã gửi bằng chứng để kiểm tra',
};
const captions = {payload: 'Xác nhận bản tổng hợp', rights: 'Quyền sử dụng', lineage: 'Nguồn gốc vật lý'};
let data, clip = 'C01', stage = 'payload', busy = false, dirty = false, selectedFiles = [], existingUploads = [];
const words = value => value === null || value === undefined || value === '' ? 'Chưa có thông tin' : (names[value] || String(value));
const number = value => Number(value).toLocaleString('vi-VN', {maximumFractionDigits: 3});
const node = (tag, text, cls) => {const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n;};
const current = () => data.clips.find(c => c.clip_id === clip);
function isoNow() {
  const d = new Date(), offset = -d.getTimezoneOffset();
  return new Date(d.getTime() + offset * 60000).toISOString().slice(0, 19) + (offset >= 0 ? '+' : '-') + String(Math.floor(Math.abs(offset) / 60)).padStart(2, '0') + ':' + String(Math.abs(offset) % 60).padStart(2, '0');
}
function preparedNotes(kind) {
  if (kind === 'payload') return 'Tôi đã đọc bản tổng hợp từ các mục Review1 đã duyệt và xác nhận nguyên trạng nội dung đang được đóng SHA256.';
  if (kind === 'rights') return 'Hiện chưa có bằng chứng đủ để quyết định quyền sử dụng video cho dữ liệu temporal hoặc calibration; giữ trạng thái chờ người có trách nhiệm và tài liệu phù hợp.';
  return 'Hiện chưa có bằng chứng xác nhận source, camera, phiên quay, xe và người vật lý; giữ NOT_PROVABLE và không tạo mã suy đoán để vượt gate.';
}
function status(text, kind = '') {$('message').textContent = text; $('message').className = kind;}
function canLeave() {return !busy && (!dirty || window.confirm('Bạn có thông tin chưa lưu. Chuyển phần và bỏ các thay đổi đang nhập?'));}
async function load() {
  $('refresh').disabled = true;
  try {
    const response = await fetch('/api/final-review');
    const value = await response.json();
    if (!response.ok) throw new Error(value.detail || 'Không tải được hồ sơ.');
    data = value; dirty = false; render();
    status('Dữ liệu đã tải. Chọn một clip và hoàn thiện từng phần; không cần duyệt lại 37 đề xuất.');
  } catch (error) {status(error.message + ' Bấm Tải lại hồ sơ để thử lại.', 'error');}
  finally {$('refresh').disabled = false;}
}
function render() {
  const good = data.clips.filter(c => !c.error);
  $('progress').textContent = `${good.filter(c => c.states.payload.submitted).length}/4 bản tổng hợp có xác nhận · ${good.filter(c => c.states.rights.submitted).length}/4 hồ sơ quyền sử dụng · ${good.filter(c => c.states.lineage.submitted).length}/4 khai báo nguồn gốc`;
  for (const kind of ['payload', 'rights', 'lineage']) {
    const done = good.filter(c => c.states[kind].submitted).length;
    $(`quick-${kind}-state`).textContent = done === 4 ? 'Đã hoàn tất 4/4' : `${done}/4 đã ghi · ${4 - done}/4 còn lại`;
    $(`quick-${kind}`).disabled = busy || done === 4 || !$(`quick-${kind}-check`).checked;
  }
  const totalDone = good.reduce((sum, c) => sum + Object.values(c.states).filter(s => s.submitted).length, 0);
  $('quick-all-state').textContent = totalDone === 12 ? 'Đã hoàn tất 12/12 hồ sơ' : `${12 - totalDone} hồ sơ đang chờ thao tác của admin`;
  $('quick-all').disabled = busy || totalDone === 12 || !$('quick-all-check').checked;
  $('clips').replaceChildren();
  for (const c of data.clips) {
    const button = node('button');
    button.append(node('strong', c.clip_id), node('small', c.error ? 'Lỗi hồ sơ' : `${Object.values(c.states).filter(s => s.submitted).length}/3 phần đã gửi`));
    button.setAttribute('aria-current', String(c.clip_id === clip));
    button.onclick = () => {if (!canLeave()) return; clip = c.clip_id; render(); $('clips').querySelector('[aria-current=true]').focus({preventScroll: true});};
    $('clips').append(button);
  }
  for (const button of $('stages').querySelectorAll('button')) {
    if (button.dataset.stage === stage) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current');
  }
  const c = current();
  $('content').hidden = !!c.error;
  if (c.error) {status(`${clip}: ${c.error}`, 'error'); return;}
  $('dossier-title').textContent = `${clip} · ${captions[stage]}`;
  $('candidate-download').href = `/api/final-review/${clip}/download/candidate`;
  renderDossier(c); renderForm(c); dirty = false;
}
function facts(parent, pairs) {
  const list = node('dl');
  for (const [label, value] of pairs) list.append(node('dt', label), node('dd', words(value)));
  parent.append(list);
}
function disclosure(parent, label, value) {
  const d = node('details'); d.append(node('summary', label), node('pre', JSON.stringify(value, null, 2), 'json-view')); parent.append(d);
}
function link(parent, label, url) {
  if (!url) return;
  try {if (!['http:', 'https:'].includes(new URL(url).protocol)) return;} catch {return;}
  const a = node('a', label, 'source-link'); a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer';
  const p = node('p'); p.append(a); parent.append(p);
}
function intervals(parent, title, rows, context = false) {
  parent.append(node('h3', title));
  const table = node('table', undefined, 'interval-table');
  const head = node('thead'), tr = node('tr');
  ['Thời gian', context ? 'Bối cảnh đề xuất' : 'Trạng thái', 'Người / quan sát'].forEach(s => tr.append(node('th', s)));
  head.append(tr); table.append(head); const body = node('tbody');
  for (const r of rows) {
    const row = node('tr'); const time = node('td', `${number(r.start_time)}–${number(r.end_time)} giây`);
    time.append(node('small', `Khung ${r.start_frame}–${r.end_frame}, không gồm mốc cuối`));
    const value = context ? (r.inside_vehicle === true ? 'Trong xe' : r.outside_vehicle_person === true ? 'Ngoài xe' : 'Trong / ngoài xe chưa rõ') : words(r.state);
    const state = node('td', value);
    if (context) state.append(node('small', `Xe máy: ${r.motorcycle_flag === null ? 'chưa rõ' : r.motorcycle_flag ? 'có' : 'không'}`), node('small', r.conditions));
    const details = node('details'); details.append(node('summary', 'Căn cứ gốc'), node('p', r.review_notes)); state.append(details);
    row.append(time, state, node('td', `${words(r.occupant_role_proposal)} · ${words(r.visibility)}`)); body.append(row);
  }
  table.append(body); const scroll = node('div', undefined, 'table-scroll'); scroll.append(table); parent.append(scroll);
}
function renderDossier(c) {
  const box = $('dossier'); box.replaceChildren(); const p = c.candidate.assembled_payload;
  const note = stage === 'payload' ? 'Bạn xác nhận bản ghép đầy đủ, gồm nội dung đã duyệt và thông tin kỹ thuật đi kèm. Không phải duyệt lại 37 item.' : stage === 'rights' ? 'Quyết định dùng dữ liệu phải dựa trên tài liệu áp dụng cho đúng video và mục đích dự án.' : 'Cần tài liệu xác định nguồn, camera, phiên quay, xe và người. Tên clip hoặc hình ảnh giống nhau không chứng minh được danh tính vật lý.';
  box.append(node('p', note, 'form-header-copy'));
  if (stage === 'payload') {
    const s = p.sequence;
    facts(box, [['Video', clip], ['Item đã duyệt', c.candidate.approved_item_receipts.length], ['Thời lượng', `${number(s.duration_seconds)} giây`], ['Số khung hình', s.frame_count], ['FPS', s.fps_rational || s.fps], ['Độ phân giải', `${s.width} × ${s.height}`], ['Định dạng', `${s.container} / ${s.codec}`]]);
    box.append(node('h3', 'Người và liên kết xe trong đề xuất'));
    for (const o of p.identity.occupants) facts(box, [['Người đề xuất', o.occupant_id_proposal], ['Vai trò', o.role_proposal], ['Xe / cabin đề xuất', `${o.vehicle_id_proposal} / ${o.cabin_id_proposal}`], ['Trong xe', o.inside_vehicle_proposal === null ? 'Chưa xác định cho toàn clip' : o.inside_vehicle_proposal ? 'Có' : 'Không']]);
    box.append(node('p', 'Các mã trên vẫn là đề xuất, chưa thay thế roster định danh độc lập.', 'help'));
    intervals(box, 'Điện thoại', s.phone_intervals); intervals(box, 'Dây an toàn', s.seatbelt_intervals); intervals(box, 'Bối cảnh toàn thời gian', s.context_intervals, true);
    box.append(node('h3', 'Các phần cần bằng chứng riêng'));
    facts(box, [['Quyền sử dụng trong bản gốc', p.rights.project_use_review], ['Nguồn gốc trong bản gốc', p.physical_lineage.lineage_status]]);
    disclosure(box, 'Xem đầy đủ bốn phần và metadata của bản tổng hợp', p);
  } else if (stage === 'rights') {
    facts(box, [['Nguồn tham chiếu', p.rights.source_page_url], ['Tác giả tham chiếu', p.rights.creator], ['Mã tài sản', p.rights.asset_id], ['Tên tệp gốc', p.rights.original_filename], ['Ngày tải có ghi nhận', p.rights.downloaded_at], ['Quyền sử dụng trong proposal', p.rights.project_use_review]]);
    link(box, 'Mở trang nguồn tham chiếu', p.rights.source_page_url);
    link(box, 'Mở trang điều khoản tham chiếu', p.rights.license_or_terms_url);
    box.append(node('h3', 'Chuẩn bị trước khi gửi'));
    const list = node('ul', undefined, 'hint-list');
    ['Tài liệu giấy phép / điều khoản hoặc chấp thuận áp dụng cho video này.', 'Căn cứ liên kết tài liệu với đúng video đã tải.', 'Mục đích sử dụng cụ thể của dự án và quyết định của người có trách nhiệm.', 'Lịch sử tải nếu có ghi nhận; nếu chưa biết, để trống.'].forEach(s => list.append(node('li', s))); box.append(list);
    box.append(node('p', 'Đường dẫn và tên tác giả chỉ là tham chiếu đã có. Tệp tải lên sẽ được lưu và tính SHA; công cụ không tự kết luận nội dung giấy phép cho phép sử dụng.', 'notice'));
  } else {
    box.append(node('h3', 'Bạn có bằng chứng về các nhóm vật lý không?'));
    box.append(node('p', 'Nếu chưa có, chọn “Chưa chứng minh được”, ghi lý do rồi lưu. Không cần điền các mã nguồn/camera/xe/người khi chọn phương án này.', 'notice'));
    facts(box, [['Trạng thái trong proposal', p.physical_lineage.lineage_status], ['Nhóm xe', p.physical_lineage.physical_vehicle_group_id_proposal], ['Phiên ghi hình', p.physical_lineage.capture_session_id_proposal]]);
    box.append(node('h3', 'Khi có tài liệu thật'));
    const list = node('ul', undefined, 'hint-list');
    ['Mã nguồn, camera và phiên ghi hình do người/tài liệu xác nhận.', 'Nhóm xe vật lý và đầy đủ nhóm người xuất hiện.', 'Nếu nhiều xe: ánh xạ từng xe được xác nhận sang nhóm xe vật lý.', 'Tài liệu, ghi chú, người xác nhận và thời điểm; các mã sẽ còn được kiểm tra độc lập.'].forEach(s => list.append(node('li', s))); box.append(list);
    box.append(node('p', 'Không suy nhóm vật lý từ C01/C07, tên tệp, URL, người đăng hoặc ngoại hình.', 'help'));
  }
  const video = node('details'); video.append(node('summary', 'Mở video gốc để đối chiếu khi cần'));
  const player = node('video'); player.controls = true; player.preload = 'none'; player.playsInline = true; player.src = `/api/video/${clip}`; video.append(player); box.append(video);
  disclosure(box, 'Mã SHA256 của nội dung đang xác nhận', {video_sha256: c.candidate.video_sha256, candidate_sha256: c.version, final_payload_sha256: c.candidate.assembled_payload_sha256, record_payload_sha256: c.candidate.assembled_record_payload_sha256});
}
function input(parent, id, label, value = '', required = false, type = 'text') {
  const l = node('label', label); l.htmlFor = id;
  const n = node(type === 'textarea' ? 'textarea' : 'input'); n.id = id; n.value = value || ''; n.required = required;
  if (type === 'textarea') n.rows = 3; else n.type = type;
  parent.append(l, n); return n;
}
function select(parent, id, label, options) {
  const l = node('label', label); l.htmlFor = id; const s = node('select'); s.id = id; s.required = true;
  for (const [value, title] of options) {const o = node('option', title); o.value = value; s.append(o);}
  parent.append(l, s); return s;
}
function renderForm(c) {
  const state = c.states[stage]; const doc = state.document;
  $('form-fields').disabled = false; $('review-form').reset();
  $('stage-fields').replaceChildren(); $('file-list').replaceChildren(); selectedFiles = [];
  existingUploads = (doc?.evidence || []).map(e => e.upload_id);
  $('form-error').textContent = ''; $('save').disabled = false;
  $('record-state').textContent = state.submitted ? 'Đã có hồ sơ gửi' : 'Chưa xác nhận';
  $('form-title').textContent = `${clip} · ${stage === 'payload' ? 'Ký bản tổng hợp' : 'Khai báo của bạn'}`;
  $('saved-record').hidden = !state.submitted; $('saved-record').replaceChildren();
  if (state.submitted) {
    const banner = node('div', undefined, 'submitted-banner');
    banner.append(node('strong', `Đã ghi: ${words(doc.decision)}`), node('p', `${doc.reviewer_id} · ${doc.reviewed_at}`), node('p', 'Hồ sơ đã được lưu; tính hợp lệ của bằng chứng và các gate chưa được tự xác nhận.'));
    const a = node('a', 'Tải hồ sơ đã gửi'); a.href = `/api/final-review/${clip}/download/${stage}`; banner.append(a); $('saved-record').append(banner);
  }
  $('upload-section').hidden = stage === 'payload';
  const target = $('stage-fields');
  if (stage === 'payload') {
    target.append(node('p', 'Đọc toàn bộ bản tổng hợp bên cạnh, đặc biệt metadata, định danh đề xuất và các khoảng thời gian. Sau đó điền xác nhận bên dưới.', 'form-header-copy'));
    target.append(node('p', 'Xác nhận nguyên trạng không đồng nghĩa quyền sử dụng được chấp nhận hoặc nguồn gốc đã được chứng minh.', 'notice'));
    const details = node('details'); details.append(node('summary', 'Nếu bản tổng hợp cần sửa?'), node('p', 'Chưa ký bản đang hiển thị. Ghi lại nội dung cần sửa để chuẩn bị một candidate riêng và tính SHA mới; không sửa trực tiếp proposal hoặc ký hash cũ.', 'help')); target.append(details);
    $('save').textContent = 'Xác nhận nguyên trạng và lưu biên nhận';
    $('attestation-copy').textContent = 'Tôi đã đọc toàn bộ bản tổng hợp của clip này, gồm metadata đi kèm, và xác nhận đúng nội dung đang đóng hash. Tôi không thay đổi các phần còn chờ quyền sử dụng hoặc nguồn gốc.';
    if (state.submitted) {$('review-form').hidden = true; return;}
  } else if (stage === 'rights') {
    const p = c.candidate.assembled_payload.rights;
    select(target, 'decision', 'Quyết định quyền sử dụng', [['', 'Chọn sau khi kiểm tra tài liệu'], ['ACCEPT', 'Chấp nhận cho mục đích nêu dưới đây'], ['REJECT', 'Từ chối sử dụng'], ['NEEDS_HUMAN_DECISION', 'Chưa đủ căn cứ để quyết định']]);
    input(target, 'source-url', 'Trang nguồn áp dụng cho video', doc?.source_page_url || p.source_page_url, true, 'url');
    input(target, 'license-url', 'Trang giấy phép / điều khoản áp dụng', doc?.license_or_terms_url || p.license_or_terms_url, true, 'url');
    input(target, 'project-use', 'Mục đích sử dụng cụ thể của dự án', doc?.project_use || '', true, 'textarea');
    input(target, 'creator', 'Tác giả / người đăng (nếu biết)', doc?.creator || p.creator);
    input(target, 'asset-id', 'Mã tài sản nguồn (nếu biết)', doc?.asset_id || p.asset_id);
    input(target, 'downloaded-at', 'Thời điểm tải có ghi nhận (không biết thì để trống)', doc?.downloaded_at || '');
    input(target, 'download-method', 'Cách tải có ghi nhận (không biết thì để trống)', doc?.download_method || '');
    $('save').textContent = state.submitted ? 'Gửi bản bổ sung quyền sử dụng' : 'Lưu quyết định và bằng chứng quyền sử dụng';
    $('attestation-copy').textContent = 'Tôi là người có trách nhiệm xác nhận quyết định sử dụng nêu trên. Tôi đã kiểm tra tài liệu và mối liên hệ với đúng video; nội dung khai báo là thông tin tôi thực sự biết.';
  } else {
    const decision = select(target, 'decision', 'Khả năng chứng minh nguồn gốc', [['', 'Chọn tình trạng thực tế'], ['NOT_PROVABLE', 'Chưa chứng minh được'], ['SUBMIT_FOR_VERIFICATION', 'Có tài liệu — gửi để kiểm tra']]);
    const fields = node('div'); fields.id = 'physical-fields'; fields.hidden = true;
    for (const [id, title] of [['source-id', 'Mã nguồn được xác nhận'], ['camera-id', 'Mã camera'], ['session-id', 'Mã phiên ghi hình'], ['vehicle-group', 'Mã nhóm xe vật lý chính']]) input(fields, id, title, '', false);
    input(fields, 'person-groups', 'Mã nhóm người — mỗi dòng một nhóm', '', false, 'textarea');
    select(fields, 'multiple-vehicles', 'Clip có nhiều xe cần ghi nhận?', [['', 'Chọn theo thông tin đã xác nhận'], ['no', 'Một xe'], ['yes', 'Nhiều xe']]);
    fields.append(node('h3', 'Ánh xạ xe sang nhóm vật lý'));
    fields.append(node('p', 'Dùng mã xe đã được xác nhận và nhóm vật lý có tài liệu. Không tự sao chép mã proposal.', 'help'));
    const mappings = node('div'); mappings.id = 'mappings'; fields.append(mappings);
    const add = node('button', 'Thêm xe'); add.type = 'button'; add.onclick = () => {addMapping(); dirty = true;}; fields.append(add); target.append(fields);
    decision.onchange = () => {
      const prove = decision.value === 'SUBMIT_FOR_VERIFICATION'; fields.hidden = !prove;
      for (const id of ['source-id', 'camera-id', 'session-id', 'vehicle-group', 'person-groups', 'multiple-vehicles']) $(id).required = prove;
      $('upload-section').hidden = !decision.value;
      if (!prove) for (const el of fields.querySelectorAll('input,textarea,select')) el.value = '';
    };
    $('save').textContent = state.submitted ? 'Gửi bản bổ sung nguồn gốc' : 'Lưu khai báo nguồn gốc';
    $('attestation-copy').textContent = 'Tôi xác nhận tình trạng nguồn gốc đã chọn. Các mã chỉ được cung cấp khi có căn cứ thực; nếu chưa chứng minh được, tôi giữ nguyên điều đó và không tạo mã để vượt điều kiện.';
  }
  $('review-form').hidden = false;
  $('reviewer').value = doc?.reviewer_id || 'admin';
  $('reviewed-at').value = doc?.reviewed_at || isoNow();
  $('notes').value = doc?.review_notes || preparedNotes(stage);
  drawFiles();
}
function addMapping() {
  const row = node('div', undefined, 'mapping-row');
  const a = node('input'), b = node('input'); a.placeholder = 'Mã xe được xác nhận'; b.placeholder = 'Nhóm xe vật lý';
  a.setAttribute('aria-label', 'Mã xe được xác nhận'); b.setAttribute('aria-label', 'Nhóm xe vật lý');
  const remove = node('button', 'Bỏ'); remove.type = 'button'; remove.onclick = () => {row.remove(); dirty = true;};
  row.append(a, b, remove); $('mappings').append(row); a.focus();
}
function drawFiles() {
  $('file-list').replaceChildren();
  $('file-summary').textContent = selectedFiles.length ? `${selectedFiles.length} tệp mới đã chọn` : 'Chưa chọn tệp mới';
  const doc = current().states[stage].document;
  for (const info of doc?.evidence || []) {
    if (!existingUploads.includes(info.upload_id)) continue;
    const li = node('li'); const a = node('a', `${info.filename} · đã lưu`); a.href = `/api/final-review/evidence/${info.upload_id}`;
    const remove = node('button', 'Bỏ khỏi lần gửi mới'); remove.type = 'button'; remove.onclick = () => {existingUploads = existingUploads.filter(x => x !== info.upload_id); dirty = true; drawFiles();};
    li.append(a, node('div', `SHA256: ${info.sha256}`), remove); $('file-list').append(li);
  }
  selectedFiles.forEach((file, index) => {
    const li = node('li', `${file.name} · ${number(file.size / 1024)} KB · chưa tải`);
    const remove = node('button', 'Bỏ'); remove.type = 'button'; remove.onclick = () => {selectedFiles.splice(index, 1); dirty = true; drawFiles();}; li.append(' ', remove); $('file-list').append(li);
  });
}
function now() {
  $('reviewed-at').value = isoNow(); dirty = true;
}
async function request(url, body, headers = {}) {
  const response = await fetch(url, {method: 'POST', headers: {'X-Review-Token': data.token, ...headers}, body});
  const result = await response.json(); if (!response.ok) throw new Error(result.detail || 'Không lưu được hồ sơ.'); return result;
}
async function quickSubmit(kind) {
  if (busy || !data) return false;
  const check = $(`quick-${kind}-check`);
  if (!check.checked) return false;
  const pending = data.clips.filter(c => !c.error && !c.states[kind].submitted);
  if (!pending.length) return true;
  busy = true; render();
  status(`Đang lưu ${pending.length} xác nhận ${captions[kind].toLowerCase()} của admin…`);
  try {
    for (const c of pending) {
      const body = {
        candidate_version: c.version,
        document_version: c.states[kind].version,
        reviewer_id: 'admin',
        reviewed_at: isoNow(),
        review_notes: preparedNotes(kind),
        attested: true,
        decision: kind === 'payload' ? 'APPROVE' : kind === 'rights' ? 'NEEDS_HUMAN_DECISION' : 'NOT_PROVABLE',
      };
      if (kind === 'rights') {
        const rights = c.candidate.assembled_payload.rights;
        Object.assign(body, {
          upload_ids: [], evidence_confirmed: false,
          source_page_url: rights.source_page_url,
          license_or_terms_url: rights.license_or_terms_url,
          project_use: 'Chưa có bằng chứng đủ để xác nhận quyền sử dụng cho dữ liệu temporal hoặc calibration; giữ trạng thái PENDING.',
          creator: rights.creator || '', asset_id: rights.asset_id || '',
          downloaded_at: '', download_method: '',
        });
      } else if (kind === 'lineage') {
        Object.assign(body, {upload_ids: [], evidence_confirmed: false, physical_ids: {}});
      }
      await request(`/api/final-review/${c.clip_id}/${kind}`, JSON.stringify(body), {'Content-Type': 'application/json'});
    }
    check.checked = false;
    dirty = false;
    await load();
    status(`${captions[kind]}: đã lưu xác nhận cho ${pending.length} clip. Không có trạng thái governance nào được tự nâng.`, 'success');
    return true;
  } catch (error) {
    status(`Chưa hoàn tất xác nhận hàng loạt: ${error.message}`, 'error');
    await load();
    return false;
  } finally {
    busy = false;
    render();
  }
}
async function quickSubmitAll() {
  if (busy || !data || !$('quick-all-check').checked) return;
  for (const kind of ['payload', 'rights', 'lineage']) {
    $(`quick-${kind}-check`).checked = true;
    if (!(await quickSubmit(kind))) return;
  }
  $('quick-all-check').checked = false;
  render();
  status('Đã lưu toàn bộ xác nhận của admin: 4 payload được xác nhận, rights giữ PENDING và lineage giữ NOT_PROVABLE.', 'success');
}
$('review-form').onsubmit = async event => {
  event.preventDefault(); if (busy) return;
  const c = current(), chosenClip = clip, chosenStage = stage;
  const body = {candidate_version: c.version, document_version: c.states[stage].version, reviewer_id: $('reviewer').value, reviewed_at: $('reviewed-at').value, review_notes: $('notes').value, attested: $('attested').checked, decision: stage === 'payload' ? 'APPROVE' : $('decision').value};
  $('form-error').textContent = '';
  try {
    if (stage !== 'payload') {
      const total = existingUploads.length + selectedFiles.length;
      if (total > 10) throw new Error('Chọn tối đa 10 tệp evidence.');
      if (selectedFiles.some(f => !f.size || f.size > 20 * 1024 * 1024)) throw new Error('Tệp phải có nội dung và không quá 20 MB.');
      if (total && !$('evidence-confirmed').checked) throw new Error('Tích xác nhận đã đọc các tệp bằng chứng.');
      if (!total && ['ACCEPT', 'REJECT', 'SUBMIT_FOR_VERIFICATION'].includes(body.decision)) throw new Error('Quyết định này cần ít nhất một tệp bằng chứng thực.');
      body.evidence_confirmed = $('evidence-confirmed').checked;
      if (stage === 'rights') {
        for (const [key, id] of [['source_page_url','source-url'],['license_or_terms_url','license-url'],['project_use','project-use'],['creator','creator'],['asset_id','asset-id'],['downloaded_at','downloaded-at'],['download_method','download-method']]) body[key] = $(id).value;
      } else {
        body.physical_ids = {};
        if (body.decision === 'SUBMIT_FOR_VERIFICATION') {
          body.physical_ids = {source_id: $('source-id').value, camera_id: $('camera-id').value, capture_session_id: $('session-id').value, physical_vehicle_group_id: $('vehicle-group').value, person_group_ids: $('person-groups').value.split('\n').map(s => s.trim()).filter(Boolean), vehicle_physical_groups: {}};
          for (const row of $('mappings').children) {const [a,b] = row.querySelectorAll('input'); if (!a.value.trim() || !b.value.trim()) throw new Error('Điền đủ hai mã cho mỗi dòng ánh xạ xe, hoặc bỏ dòng trống.'); if (Object.hasOwn(body.physical_ids.vehicle_physical_groups, a.value.trim())) throw new Error('Mã xe bị lặp trong bảng ánh xạ.'); body.physical_ids.vehicle_physical_groups[a.value.trim()] = b.value.trim();}
          body.multiple_vehicles = $('multiple-vehicles').value === 'yes';
        }
      }
    }
    busy = true; $('form-fields').disabled = true; $('refresh').disabled = true;
    status('Đang kiểm tra và lưu xác nhận của bạn…');
    if (stage !== 'payload') {
      body.upload_ids = [...existingUploads];
      for (const file of selectedFiles) {const info = await request(`/api/final-review/${chosenClip}/evidence/${chosenStage}`, file, {'Content-Type': 'application/octet-stream', 'X-File-Name': encodeURIComponent(file.name)}); body.upload_ids.push(info.upload_id);}
    }
    const result = await request(`/api/final-review/${chosenClip}/${chosenStage}`, JSON.stringify(body), {'Content-Type': 'application/json'});
    dirty = false; await load(); status(`${chosenClip}: ${result.message}`, 'success');
  } catch (error) {$('form-error').textContent = error.message; status('Chưa lưu xác nhận. Kiểm tra thông báo trong biểu mẫu.', 'error');}
  finally {busy = false; $('form-fields').disabled = false; $('refresh').disabled = false;}
};
$('evidence').onchange = () => {selectedFiles.push(...$('evidence').files); $('evidence').value = ''; dirty = true; drawFiles();};
$('review-form').addEventListener('input', () => {dirty = true;});
$('now').onclick = now;
$('refresh').onclick = () => {if (canLeave()) load();};
for (const kind of ['payload', 'rights', 'lineage']) {
  $(`quick-${kind}-check`).onchange = () => render();
  $(`quick-${kind}`).onclick = () => quickSubmit(kind);
}
$('quick-all-check').onchange = () => render();
$('quick-all').onclick = quickSubmitAll;
for (const button of $('stages').querySelectorAll('button')) button.onclick = () => {if (!data || !canLeave()) return; stage = button.dataset.stage; render(); button.focus({preventScroll: true});};
window.addEventListener('beforeunload', event => {if (dirty) {event.preventDefault(); event.returnValue = '';}});
load();
