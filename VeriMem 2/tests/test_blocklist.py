from retrieval.blocklist import domain_of, filter_results, filter_urls, is_blocked


def test_domain_strips_www_and_port():
    assert domain_of("https://www.Example.org:8080/a/b") == "example.org"


def test_known_fact_checkers_blocked():
    for url in [
        "https://www.politifact.com/factchecks/2024/x/",
        "http://snopes.com/fact-check/y",
        "https://fullfact.org/health/z",
    ]:
        assert is_blocked(url)


def test_subdomains_blocked():
    assert is_blocked("https://amp.politifact.com/a")


def test_substring_rule_catches_unlisted_factcheck_hosts():
    assert is_blocked("https://factcheck.example.com/a")
    assert is_blocked("https://fact-check.somenews.co.uk/a")


def test_ordinary_sources_pass():
    for url in [
        "https://en.wikipedia.org/wiki/X",
        "https://nature.com/articles/1",
        "https://bbc.co.uk/news/1",
    ]:
        assert not is_blocked(url)


def test_malformed_url_is_blocked():
    assert is_blocked("")
    assert is_blocked("not a url")


def test_filters():
    urls = ["https://snopes.com/a", "https://wikipedia.org/b"]
    assert filter_urls(urls) == ["https://wikipedia.org/b"]
    results = [{"url": u, "text": "t"} for u in urls]
    assert filter_results(results) == [{"url": "https://wikipedia.org/b", "text": "t"}]


# --- archive wrappers (rule 5 hazard: nearly every AVeriTeC URL is one) -------------


def test_unwraps_a_wayback_url_to_the_publisher():
    url = "https://web.archive.org/web/20201129141238/https://www.nbcnews.com/politics/x"
    assert domain_of(url) == "nbcnews.com"


def test_a_wayback_snapshot_of_a_fact_checker_is_blocked():
    """The whole point: the literal host is web.archive.org, which is not blocked."""
    assert is_blocked("https://web.archive.org/web/2020/https://www.politifact.com/x")
    assert is_blocked("https://web.archive.org/web/2019id_/http://snopes.com/fact-check/y")


def test_a_wayback_snapshot_of_an_ordinary_source_passes():
    assert not is_blocked("https://web.archive.org/web/2020/https://en.wikipedia.org/wiki/X")


def test_plain_archive_url_without_an_embedded_target():
    """archive.ph mints opaque ids; there is nothing to unwrap, so it stays itself."""
    assert domain_of("https://archive.ph/2Cpq5") == "archive.ph"


def test_bare_archive_root_is_not_misread():
    assert domain_of("https://web.archive.org/") == "web.archive.org"


def test_unwrap_returns_none_when_there_is_no_inner_url():
    from retrieval.blocklist import unwrap_archive

    assert unwrap_archive("https://example.org/a/b") is None


def test_trust_domains_differ_across_wayback_sources():
    """C3 needs real publishers; otherwise every source shares one domain."""
    a = domain_of("https://web.archive.org/web/2020/https://bbc.co.uk/news/1")
    b = domain_of("https://web.archive.org/web/2020/https://nature.com/articles/2")
    assert a == "bbc.co.uk" and b == "nature.com" and a != b
