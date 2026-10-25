/// A connection URL with its password replaced by `***`, safe to log or put in
/// an error message. `redis://:secret@host` and `postgres://u:secret@host/db`
/// both carry credentials in the userinfo part; a URL without one is unchanged.
pub fn redact_url(url: &str) -> String {
    let Some(scheme_end) = url.find("://").map(|i| i + 3) else {
        return url.to_string();
    };
    let rest = &url[scheme_end..];
    // The last `@` ends the userinfo: an unescaped password may itself hold `@` or `/`.
    let Some(at) = rest.rfind('@') else {
        return url.to_string();
    };
    match rest[..at].find(':') {
        Some(colon) => format!(
            "{}{}:***{}",
            &url[..scheme_end],
            &rest[..colon],
            &rest[at..]
        ),
        None => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_password_in_a_url_is_masked() {
        assert_eq!(
            redact_url("postgres://flux:s3cret@db:5432/flux"),
            "postgres://flux:***@db:5432/flux"
        );
        assert_eq!(
            redact_url("redis://:s3cret@cache:6379/0"),
            "redis://:***@cache:6379/0"
        );
    }

    #[test]
    fn a_password_containing_at_or_slash_is_masked_whole() {
        assert_eq!(
            redact_url("redis://u:p@ss/w@cache:6379"),
            "redis://u:***@cache:6379"
        );
    }

    #[test]
    fn a_url_without_a_password_is_unchanged() {
        assert_eq!(
            redact_url("redis://127.0.0.1:6379"),
            "redis://127.0.0.1:6379"
        );
        assert_eq!(
            redact_url("postgres://flux@db/flux"),
            "postgres://flux@db/flux"
        );
        assert_eq!(redact_url("not a url"), "not a url");
    }
}
