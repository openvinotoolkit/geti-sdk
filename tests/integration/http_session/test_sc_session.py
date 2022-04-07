class TestSCSession:
    def test_authenticate(self, fxt_sc_session):
        """
        Test that the authenticated SCSession instance contains authentication cookies
        """
        for cookie_name, cookie_value in fxt_sc_session._cookies.items():
            assert cookie_value is not None

    def test_product_version(self, fxt_sc_session):
        """
        Test that the 'version' attribute of the session is assigned a valid product
        version
        """
        assert fxt_sc_session.version in ['1.0', '1.1']
