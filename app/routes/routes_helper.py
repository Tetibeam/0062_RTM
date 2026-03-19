from flask import (
    request,
    make_response,
    jsonify,
    current_app
)
from typing import Dict, Any

import json
import hashlib

def apply_etag(payload: dict):
    """
    レスポンスボディとリクエストヘッダーを基にETag処理を行うヘルパー関数。
    
    Args:
        payload (dict): レスポンスとして返すデータ（辞書）。
        
    Returns:
        Response: ETagヘッダーが付与されたResponseオブジェクト、
                  または 304 Not Modified Response。
    """
    
    # 1. データのハッシュ化（ETagの生成）
    # JSONのバイト列を生成し、SHA1でハッシュ化します。
    # ensure_ascii=False と sort_keys=True は、常に同じハッシュが得られるように重要です。
    json_string = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    current_etag = hashlib.sha1(json_string.encode('utf-8')).hexdigest()
    
    # クライアントが前回送信したETagを取得
    # request.headers.get('If-None-Match') は、クライアントが持つETagです
    client_etag = request.headers.get('If-None-Match')
    
    # 2. ヘッダーの確認と 3. 条件付きレスポンス
    if client_etag and client_etag == current_etag:
        # ETagが一致する場合 (データに変更なし)
        resp = make_response('', 304)
        # 304を返す場合でも、クライアントに最新のETagを再送する慣習があります
        resp.headers['ETag'] = current_etag
        current_app.logger.info(f"ETag matched: 304 Not Modified for {request.path}")
        return resp

    # ETagが不一致または初回リクエストの場合 (データを返す必要がある)
    resp = make_response(jsonify(payload), 200)
    resp.headers['ETag'] = current_etag
    
    # キャッシュを厳しく制御し、次回必ずAPIを叩くように設定
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    
    return resp

# ----------------------------------------------------------------------
# カスタムキャッシュキー生成関数
# ----------------------------------------------------------------------

def key_generator_with_params(graph_id: str, params: Dict[str, Any]):
    # 1. params 辞書をソートし、JSON文字列に変換
    try:
        # paramsの内容が変われば、この文字列は必ず変わる
        params_str = json.dumps(params, sort_keys=True)
    except Exception:
        params_str = str(params)
    
    # 2. 最終キーを生成: モジュール名を含めず、関数名と引数のみでキーを作成
    #    この関数が使われる場所は限定的であるため、これで衝突は起きにくい
    cache_key = f'{graph_id}:{params_str}'
    
    return cache_key

# ----------------------------------------------------------------------