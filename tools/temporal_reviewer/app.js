'use strict';
const $ = (id) => document.getElementById(id);
const labels = {
  rights:'Quyền sử dụng', physical_lineage:'Nguồn gốc vật lý', identity:'Định danh người và xe',
  CONTEXT:'Bối cảnh', UNKNOWN:'Chưa xác định', PHONE_USE:'Đang sử dụng điện thoại',
  PHONE_PRESENT_NOT_USED:'Có điện thoại, chưa sử dụng', MOUNTED_OR_STATIC_PHONE:'Điện thoại gắn cố định',
  NO_PHONE:'Không có điện thoại', FASTENED:'Đã cài dây an toàn', UNFASTENED:'Chưa cài dây an toàn',
  UNCERTAIN_OR_OCCLUDED:'Chưa rõ hoặc bị che khuất', NOT_APPLICABLE:'Không áp dụng',
  driver:'Người lái', passenger:'Hành khách', unknown:'Chưa rõ', clear:'Rõ', partial:'Một phần',
  occluded:'Bị che khuất', out_of_view:'Ngoài khung hình', APPROVE:'Đã phê duyệt',
  EDIT_AND_APPROVE:'Đã sửa và phê duyệt', REJECT:'Đã từ chối', UNCERTAIN:'Chưa chắc chắn', PENDING:'Đang chờ',
};
const humanFields = ['HUMAN_DECISION','REVIEWER_ID','REVIEWED_AT','HUMAN_NOTES','EDITED_ITEM_JSON','REVIEW_EVIDENCE_PATH','REVIEW_EVIDENCE_SHA256'];
let data, clip, selected, submission, busy=false;
const isPending = (r) => humanFields.every(k => !(r[k] || '').trim());
const title = (r) => labels[r.proposed_label] || r.proposed_label;
const text = (s) => labels[s] || s || 'Chưa có thông tin';
const fmt = (s) => Number(s).toLocaleString('vi-VN',{maximumFractionDigits:3});
function el(tag, value, cls) {const node=document.createElement(tag);if(value!==undefined)node.textContent=value;if(cls)node.className=cls;return node;}
function message(value, kind='') {$('message').textContent=value;$('message').className=kind;}
async function load() {
  $('reload').disabled=true;
  try {
    const response=await fetch('/api/queue');const next=await response.json();
    if(!response.ok)throw new Error(next.detail || 'Không thể tải hàng đợi.');
    data=next;clip=clip || data.items[0]?.clip_id;
    selected=data.items.find(r=>r.item_id===selected?.item_id) || data.items.find(r=>r.clip_id===clip);
    render();message('Đã tải dữ liệu local. Chưa có quyết định nào được ghi chỉ bằng việc mở trang.');
  } catch(error) {message(error.message+' Bạn có thể bấm Tải lại hàng đợi.','error');}
  finally {$('reload').disabled=false;}
}
function render() {
  const pending=data.items.filter(isPending).length;
  $('progress').textContent=`${data.items.length-pending}/${data.items.length} mục có dữ liệu người duyệt · ${pending} mục đang chờ`;
  $('clips').replaceChildren();
  for(const id of [...new Set(data.items.map(r=>r.clip_id))]) {
    const rows=data.items.filter(r=>r.clip_id===id);const button=el('button');
    button.append(el('strong',id),el('small',`${rows.filter(isPending).length}/${rows.length} chờ`));
    button.setAttribute('aria-current',String(id===clip));
    button.onclick=()=>{clip=id;selected=rows.find(isPending)||rows[0];render();$('clips').querySelector('[aria-current="true"]').focus({preventScroll:true});};$('clips').append(button);
  }
  const source=`/api/video/${encodeURIComponent(clip)}`;
  if($('video').getAttribute('src')!==source){$('video-meta').textContent='Đang tải video…';$('video').src=source;$('video').playbackRate=Number($('speed').value);}
  $('video-title').textContent=`Video gốc · ${clip}`;
  const rows=data.items.filter(r=>r.clip_id===clip);
  $('clip-count').textContent=`${rows.length} mục`;
  $('items').replaceChildren();
  for(const row of rows.filter(r=>$('filter').value==='all'||($('filter').value==='pending'?isPending(r):!isPending(r)))) {
    const button=el('button',undefined,'item');button.setAttribute('aria-pressed',String(row.item_id===selected?.item_id));
    const top=el('div',undefined,'item-top');
    top.append(el('strong',title(row)),el('span',isPending(row)?'Đang chờ':text(row.HUMAN_DECISION)||'Đang nhập','tag'+(row.HUMAN_DECISION?' decided':'')));
    button.append(top,el('small',`${row.item_id} · ${row.proposed_start_seconds!==''?`${fmt(row.proposed_start_seconds)}–${fmt(row.proposed_end_seconds)} giây`:'Toàn hồ sơ'}`));
    button.onclick=()=>{selected=row;render();if(matchMedia('(max-width:760px)').matches){$('detail').focus({preventScroll:true});$('detail').scrollIntoView({block:'start'});}else{$('items').querySelector('[aria-pressed="true"]').focus({preventScroll:true});}};$('items').append(button);
  }
  if(!$('items').children.length)$('items').append(el('p','Không có mục nào trong bộ lọc này.','detail'));
  $('approve-clip').disabled=!rows.some(isPending);$('approve-all').disabled=!pending;
  renderDetail();
}
function renderDetail() {
  const box=$('detail');box.replaceChildren();if(!selected)return;
  const r=selected;const item=JSON.parse(r.proposed_item_json);
  box.append(el('h3',title(r)));
  const dl=el('dl');
  const facts=[['Mã mục',r.item_id],['Khoảng thời gian',r.proposed_start_seconds!==''?`${fmt(r.proposed_start_seconds)} → ${fmt(r.proposed_end_seconds)} giây`:'Toàn hồ sơ'],['Khung hình',r.proposed_start_frame!==''?`${r.proposed_start_frame} → ${r.proposed_end_frame_exclusive} (không gồm mốc cuối)`:'Không áp dụng'],['Vai trò đề xuất',text(r.occupant_role_proposal)],['Khả năng quan sát',text(r.VISIBILITY)],['Độ tin cậy của AI',r.CONFIDENCE?fmt(r.CONFIDENCE):'Chưa có thông tin']];
  for(const [key,value]of facts)dl.append(el('dt',key),el('dd',value));box.append(dl);
  if(r.record_type==='rights')box.append(el('p','Cần người có trách nhiệm xác nhận quyền sử dụng và cung cấp bằng chứng. Chấp nhận đề xuất đang chờ không đồng nghĩa có quyền sử dụng.','notice'));
  if(r.record_type==='physical_lineage')box.append(el('p','Chưa chứng minh được định danh nguồn, camera, phiên ghi hình, xe và người thật. Không suy từ tên clip hoặc hình ảnh.','notice'));
  const details=el('details');details.append(el('summary','Nội dung đề xuất gốc và ghi chú AI'),el('p',r.review1_notes),el('pre',JSON.stringify(item,null,2)));box.append(details);
  const provenance=el('details');provenance.append(el('summary','Nguồn và mã kiểm tra SHA256'),el('pre',`Video: ${r.video_sha256}\nHồ sơ: ${r.record_sha256}\nNội dung: ${r.payload_sha256}`));box.append(provenance);
  const actions=el('div',undefined,'decisions');
  for(const [decision,label,cls]of [['APPROVE','Phê duyệt','primary'],['EDIT_AND_APPROVE','Sửa và phê duyệt',''],['UNCERTAIN','Chưa chắc chắn',''],['REJECT','Từ chối','danger']]) {
    const button=el('button',label,cls);button.disabled=!isPending(r);button.onclick=()=>confirm([r],decision);actions.append(button);
  }
  box.append(actions);
  if(!isPending(r))box.append(el('p',`Dữ liệu đã ghi: ${text(r.HUMAN_DECISION)} · ${r.REVIEWER_ID} · ${r.REVIEWED_AT}. Công cụ không ghi đè quyết định hiện có.`));
  $('jump').disabled=r.proposed_start_seconds==='';
}
function confirm(rows, decision) {
  if(busy||!rows.length)return;
  submission={rows,decision,version:data.version};$('form-error').textContent='';
  $('confirmation-title').textContent=rows.length>1?'Xác nhận phê duyệt hàng loạt':'Xác nhận quyết định';
  const action={APPROVE:'Phê duyệt',EDIT_AND_APPROVE:'Sửa và phê duyệt',REJECT:'Từ chối',UNCERTAIN:'Giữ chưa chắc chắn'}[decision];
  $('scope').textContent=`${rows.length} mục · ${[...new Set(rows.map(r=>r.clip_id))].join(', ')} · Quyết định sẽ lưu: ${action}`;
  $('scope-items').replaceChildren(...rows.map(r=>el('div',`${r.item_id} — ${title(r)}`)));
  $('attested').checked=false;$('notes').value='';$('reviewed-at').value='';
  $('edit-wrap').hidden=decision!=='EDIT_AND_APPROVE';
  $('edited').value=decision==='EDIT_AND_APPROVE'?JSON.stringify(JSON.parse(rows[0].proposed_item_json),null,2):'';
  $('confirmation').showModal();
}
$('decision-form').onsubmit=async(event)=>{
  event.preventDefault();if(busy)return;busy=true;$('save').disabled=true;$('form-error').textContent='';
  try {
    const edited=submission.decision==='EDIT_AND_APPROVE'?JSON.parse($('edited').value):null;
    const body={version:submission.version,reviewer_id:$('reviewer').value,reviewed_at:$('reviewed-at').value,notes:$('notes').value,attested:$('attested').checked,items:submission.rows.map(r=>({item_id:r.item_id,decision:submission.decision,edited_item:edited}))};
    const response=await fetch('/api/decisions',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':data.token},body:JSON.stringify(body)});
    const result=await response.json();if(!response.ok)throw new Error(result.detail||'Không lưu được. Hãy tải lại hàng đợi.');
    $('confirmation').close();await load();message(`Đã lưu ${result.saved} quyết định và biên nhận của người duyệt. Các điều kiện governance không tự thay đổi.`,'success');
  }catch(error){$('form-error').textContent=error instanceof SyntaxError?'Nội dung JSON chưa hợp lệ. Kiểm tra lại bản sửa.':error.message;}
  finally{busy=false;$('save').disabled=false;}
};
$('time-now').onclick=()=>{const d=new Date();const offset=-d.getTimezoneOffset();const local=new Date(d.getTime()+offset*60000).toISOString().slice(0,19);$('reviewed-at').value=local+(offset>=0?'+':'-')+String(Math.floor(Math.abs(offset)/60)).padStart(2,'0')+':'+String(Math.abs(offset)%60).padStart(2,'0');};
for(const id of ['cancel','close-dialog'])$(id).onclick=()=>{if(!busy)$('confirmation').close();};
$('confirmation').addEventListener('cancel',event=>{if(busy)event.preventDefault();});
$('reload').onclick=load;$('filter').onchange=()=>{if(data)render();};
$('approve-clip').onclick=()=>confirm(data.items.filter(r=>r.clip_id===clip&&isPending(r)),'APPROVE');
$('approve-all').onclick=()=>confirm(data.items.filter(isPending),'APPROVE');
$('jump').onclick=()=>{$('video').pause();$('video').currentTime=Number(selected.proposed_start_seconds);};
$('speed').onchange=()=>{$('video').playbackRate=Number($('speed').value);};
function step(direction){const rows=data?.items.filter(r=>r.clip_id===clip)||[];const r=rows.find(r=>Number(r.proposed_end_seconds)>0&&Number(r.proposed_end_frame_exclusive)>0);const fps=r?Number(r.proposed_end_frame_exclusive)/Number(r.proposed_end_seconds):30;$('video').pause();$('video').currentTime=Math.max(0,Math.min($('video').duration||0,$('video').currentTime+direction/fps));}
$('previous-frame').onclick=()=>step(-1);$('next-frame').onclick=()=>step(1);
$('video').ontimeupdate=()=>{$('clock').textContent=`${fmt($('video').currentTime)} giây`;};
$('video').onloadedmetadata=()=>{$('video-meta').textContent=`${$('video').videoWidth} × ${$('video').videoHeight} · ${fmt($('video').duration)} giây`;};
$('video').onerror=()=>message('Không đọc được video. Kiểm tra tệp local và định dạng trình duyệt hỗ trợ.','error');
load();
