import { expect, test } from "@playwright/test";

// A full-stack regression: chat against the real (tiny) model, then verify the conversation is
// saved to history, survives a reload, and can be reopened. Exercises the React app, the
// streaming API, and the SQLite session store together.
test("chat is answered, saved to history, and survives a reload", async ({ page }) => {
  const marker = `e2e probe ${Date.now()}`;

  await page.goto("/");
  await expect(page.getByText("CRUCIBLE")).toBeVisible();

  // Send a message and wait for a streamed assistant reply.
  await page.locator(".composer textarea").fill(marker);
  await page.getByTitle("Send").click();
  await expect(page.locator(".msg.assistant .text")).not.toBeEmpty();

  // The conversation shows up in the Chats sidebar, titled from the first message.
  const chatRow = page.locator(".chat-title", { hasText: marker });
  await expect(chatRow).toBeVisible();

  // Reload: history is persisted server-side, so the chat is still listed.
  await page.reload();
  await expect(page.getByText("CRUCIBLE")).toBeVisible();
  const persisted = page.locator(".chat-title", { hasText: marker });
  await expect(persisted).toBeVisible();

  // The fresh page starts with an empty conversation; reopening restores the messages.
  await expect(page.locator(".msg.user .text")).toHaveCount(0);
  await persisted.click();
  await expect(page.locator(".msg.user .text", { hasText: marker })).toBeVisible();
  await expect(page.locator(".msg.assistant .text")).not.toBeEmpty();
});
