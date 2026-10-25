//! Local HTTP servers for backend tests.

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri};
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use serde_json::Value;
use std::sync::{Arc, Mutex};

pub struct Served {
    pub url: String,
    /// JSON body of the most recent request.
    pub last_body: Arc<Mutex<Option<Value>>>,
    /// Headers of the most recent request.
    pub last_headers: Arc<Mutex<Option<HeaderMap>>>,
    /// Path of the most recent request.
    pub last_path: Arc<Mutex<Option<String>>>,
}

/// Serves a fixed response to every POST. Must be called inside a Tokio runtime.
pub fn serve(status: u16, headers: Vec<(&str, String)>, body: &'static str) -> Served {
    let last_body = Arc::new(Mutex::new(None));
    let recorded = last_body.clone();
    let last_headers = Arc::new(Mutex::new(None));
    let recorded_headers = last_headers.clone();
    let last_path = Arc::new(Mutex::new(None));
    let recorded_path = last_path.clone();
    let headers: Vec<(HeaderName, HeaderValue)> = headers
        .into_iter()
        .filter_map(|(k, v)| Some((k.parse().ok()?, v.parse().ok()?)))
        .collect();
    let app = Router::new().fallback(post(
        move |uri: Uri, req_headers: HeaderMap, bytes: axum::body::Bytes| {
            let recorded = recorded.clone();
            let recorded_headers = recorded_headers.clone();
            let recorded_path = recorded_path.clone();
            let headers = headers.clone();
            async move {
                if let Ok(mut slot) = recorded.lock() {
                    *slot = serde_json::from_slice(&bytes).ok();
                }
                if let Ok(mut slot) = recorded_headers.lock() {
                    *slot = Some(req_headers);
                }
                if let Ok(mut slot) = recorded_path.lock() {
                    *slot = Some(uri.path().to_owned());
                }
                let mut resp = Response::new(Body::from(body));
                *resp.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
                for (k, v) in headers {
                    resp.headers_mut().insert(k, v);
                }
                resp
            }
        },
    ));
    Served {
        url: spawn(app),
        last_body,
        last_headers,
        last_path,
    }
}

/// Sends 200 headers and then never a byte of body.
pub fn serve_stall() -> String {
    let app = Router::new().fallback(post(|| async {
        let mut resp = Response::new(Body::from_stream(futures_util::stream::pending::<
            Result<axum::body::Bytes, std::convert::Infallible>,
        >()));
        resp.headers_mut().insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        resp
    }));
    spawn(app)
}

fn spawn(app: Router) -> String {
    let std_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test server");
    std_listener.set_nonblocking(true).expect("nonblocking");
    let addr = std_listener.local_addr().expect("local addr");
    let listener = tokio::net::TcpListener::from_std(std_listener).expect("tokio listener");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}")
}
