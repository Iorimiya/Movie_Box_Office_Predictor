<?php
header('Content-Type: application/json; charset=utf-8');

/**
 * Dummy 函式（之後可替換為 Python / ML / DB）
 */
function dummy_predict_boxoffice($movie) {
  return [
    "ok" => true,
    "movie" => $movie,
    "pr_class" => "pr50-80",
    "stars" => 2,
    "trace" => "dummy_predict_boxoffice() 已在 PHP 中執行"
  ];
}

$action = $_GET["action"] ?? "";

if ($action === "dummy") {
  $movie = trim($_POST["movie"] ?? "");

  if ($movie === "") {
    echo json_encode([
      "ok" => false,
      "error" => "movie is required"
    ], JSON_UNESCAPED_UNICODE);
    exit;
  }

  echo json_encode(
    dummy_predict_boxoffice($movie),
    JSON_UNESCAPED_UNICODE
  );
  exit;
}

echo json_encode([
  "ok" => false,
  "error" => "unknown action"
], JSON_UNESCAPED_UNICODE);
