"""Participant display contains only attributed, validated human reasons."""

HTML = """<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>議論の記録</title>
<style>body{font:20px system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#17212b}
h1{font-size:28px}article{border:1px solid #ccd3db;border-radius:8px;padding:16px;margin:16px 0}
small{font-size:15px;color:#53616e}blockquote{margin:10px 0;line-height:1.7}#status{font-size:16px;color:#53616e}</style>
<h1 id="title">議論</h1><p id="status"></p><main id="records"></main>
<script>
async function refresh(){try{
const r=await fetch('/api/study');if(!r.ok)return;const s=await r.json();
document.getElementById('title').textContent=s.title;
document.getElementById('status').textContent=s.condition==='system'?'会話に出た理由の記録です。誤りがあれば発言して訂正できます。':'';
const root=document.getElementById('records');root.replaceChildren();
for(const x of s.records){const a=document.createElement('article');
const label=document.createElement('strong');label.textContent=x.option;
const q=document.createElement('blockquote');q.textContent=x.quote;
const src=document.createElement('small');src.textContent=x.speaker+' ／ '+x.turn_id;
a.append(label,q,src);root.append(a);}
}catch(e){document.getElementById('status').textContent='表示の更新を待っています。';}}
refresh();setInterval(refresh,1000);
</script></html>"""
