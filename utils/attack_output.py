class attack_output():
    def __init__(self,attack_result,attack_result2,adv_sent,org_sent,adv_tr,org_tr,error_rate,adv_bleu,adv_chrf,query,ground_truth):
        self.attack_result = attack_result
        self.attack_result2 = attack_result2
        self.adv_sent=adv_sent
        self.org_sent=org_sent
        self.adv_tr=adv_tr
        self.org_tr=org_tr
        self.error_rate=error_rate
        self.adv_bleu=adv_bleu
        self.adv_chrf = adv_chrf
        self.query = query
        self.ground_truth = ground_truth
