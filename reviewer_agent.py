"""
Wind Turbine Expert LLM Reviewer Agent

This agent reviews outputs from other agents at critical checkpoints to validate
physical feasibility, catch workflow errors, and provide expert feedback.

Checkpoints:
- Checkpoint 1: After Agent 2 variants (weather-based yaw/pair recommendations)
- Checkpoint 2: After Agent 3 (power predictions and optimization)
- Checkpoint 3: After Agent 4 (wake flow predictions)

Reviewer Modes:
- Advisory: Provides feedback but never blocks workflow
- Blocking: Halts workflow on critical issues requiring user attention
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import LLM infrastructure
from llm.base import LLMFactory

logger = logging.getLogger(__name__)


class WindTurbineReviewerAgent:
    """
    Expert LLM reviewer that validates agent outputs at critical workflow checkpoints.
    Provides domain expertise on wind turbine operations, wake effects, and power predictions.
    """
    
    # NREL 5MW Turbine Specifications
    TURBINE_SPECS = {
        "model": "NREL 5MW",
        "rated_power_mw": 5.0,
        "rotor_diameter_m": 126.0,
        "hub_height_m": 90.0,
        "cut_in_speed_ms": 3.0,
        "cut_out_speed_ms": 25.0,
        "rated_wind_speed_ms": 11.4,
        "yaw_range_deg": (272, 285),
        "max_yaw_misalignment_deg": 15.0
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = "advisory",
        enabled: bool = True
    ):
        """
        Initialize the reviewer agent.
        
        Args:
            config: Configuration dict from config.yaml
            mode: "advisory" or "blocking"
            enabled: Whether reviewer is enabled
        """
        self.config = config
        self.mode = mode.lower()
        self.enabled = enabled
        
        if self.mode not in ["advisory", "blocking"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'advisory' or 'blocking'")
        
        # Initialize LLM from config
        self.llm = None
        if self.enabled:
            try:
                reviewer_config = config.get("reviewer", {})
                use_same_provider = reviewer_config.get("use_same_provider_as_agent2b", True)
                
                if use_same_provider:
                    # Import global config and use global LLM configuration
                    try:
                        from wind_turbine_gui import GlobalLLMConfig
                        # Use global configuration if available
                        provider = GlobalLLMConfig.get_provider().lower()
                        model = GlobalLLMConfig.get_model()
                        api_base, api_key = GlobalLLMConfig.get_api_config()
                    except:
                        # Fallback to config file
                        llm_config = config.get("llm", {})
                        provider = "ntnu"  # Default provider
                        model = llm_config.get("model", "moonshotai/Kimi-K2.5")
                        api_key = llm_config.get("api_key", "")
                        api_base = llm_config.get("api_base", "https://llm.hpc.ntnu.no/v1")
                else:
                    # Use dedicated reviewer config
                    provider = reviewer_config.get("provider", "ntnu")
                    model = reviewer_config.get("model", "moonshotai/Kimi-K2.5")
                    api_key = reviewer_config.get("api_key", "")
                    api_base = reviewer_config.get("api_base", "")
                
                self.temperature = reviewer_config.get("temperature", 0.3)
                self.max_tokens = reviewer_config.get("max_tokens", 1500)
                
                # Create LLM instance
                self.llm = LLMFactory.create(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=api_base,
                    timeout=config.get("llm", {}).get("timeout", 300.0)
                )
                
                logger.info(f"Reviewer initialized with {provider} provider, mode={self.mode}")
                
            except Exception as e:
                logger.error(f"Failed to initialize LLM for reviewer: {e}")
                self.llm = None
        
        # Storage for all reviews
        self.checkpoint_reviews: List[Dict[str, Any]] = []
    
    async def review_agent2(
        self,
        weather_data: Dict[str, Any],
        expert_analysis: Dict[str, Any],
        agent2b_result: Optional[Dict[str, Any]] = None,
        turbine_pairs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Review Agent 2 outputs (weather-based recommendations).
        
        Validates:
        - Wind speed within cut-in/cut-out range
        - Yaw angle recommendations validity
        - Operating region determination
        - Agent 2B consistency (if used)
        - Turbine pair alignment (if Agent 2C/2D used)
        
        Args:
            weather_data: Output from Weather Agent
            expert_analysis: Output from NREL Expert Agent
            agent2b_result: Optional output from Agent 2B (LLM expert)
            turbine_pairs: Optional output from Agent 2C/2D (pair selector)
        
        Returns:
            Review dict with findings, severity, and allow_continue flag
        """
        if not self.enabled:
            return self._disabled_review("checkpoint1_agent2")
        
        logger.info("Starting Checkpoint 1: Agent 2 Review")
        
        # Rule-based validation checks
        findings = []
        severity = "info"
        
        # Extract wind speed
        wind_speed = weather_data.get("wind_speed_ms", 0)
        wind_direction = weather_data.get("wind_direction_deg", 0)
        
        # Check 1: Wind speed cut-off validation
        if wind_speed < self.TURBINE_SPECS["cut_in_speed_ms"]:
            findings.append({
                "type": "critical",
                "message": f"Wind speed {wind_speed:.2f} m/s is BELOW cut-in speed ({self.TURBINE_SPECS['cut_in_speed_ms']} m/s). Turbine should be parked, optimization not recommended.",
                "rule": "cut_in_violation"
            })
            severity = "critical"
        elif wind_speed > self.TURBINE_SPECS["cut_out_speed_ms"]:
            findings.append({
                "type": "critical",
                "message": f"Wind speed {wind_speed:.2f} m/s is ABOVE cut-out speed ({self.TURBINE_SPECS['cut_out_speed_ms']} m/s). Turbine should be shut down for safety.",
                "rule": "cut_out_violation"
            })
            severity = "critical"
        elif wind_speed < 4.0:
            findings.append({
                "type": "warning",
                "message": f"Wind speed {wind_speed:.2f} m/s is marginal (just above cut-in). Power generation will be minimal.",
                "rule": "marginal_wind_speed"
            })
            if severity == "info":
                severity = "warning"
        
        # Check 2: Yaw angle validation
        suggested_yaw = expert_analysis.get("suggested_yaw", 0)
        yaw_min, yaw_max = self.TURBINE_SPECS["yaw_range_deg"]
        
        if not (yaw_min <= suggested_yaw <= yaw_max):
            findings.append({
                "type": "warning",
                "message": f"Suggested yaw angle {suggested_yaw:.1f}° is outside typical range ({yaw_min}-{yaw_max}°). Verify feasibility.",
                "rule": "yaw_range_check"
            })
            if severity == "info":
                severity = "warning"
        
        # Check 3: Operating region consistency
        operating_region = expert_analysis.get("operating_region", "Unknown")
        if wind_speed < self.TURBINE_SPECS["cut_in_speed_ms"] and operating_region != "Below Cut-in":
            findings.append({
                "type": "critical",
                "message": f"Operating region mismatch: Wind speed indicates 'Below Cut-in' but agent reports '{operating_region}'",
                "rule": "operating_region_consistency"
            })
            severity = "critical"
        
        # Check 4: Turbine pair validation (if applicable)
        if turbine_pairs and turbine_pairs.get("status") == "success":
            pairs = turbine_pairs.get("turbine_pairs", [])
            for pair in pairs:
                if not pair.get("validated", True):
                    findings.append({
                        "type": "warning",
                        "message": f"Turbine pair {pair.get('upstream_turbine')}->{pair.get('downstream_turbine')} has alignment deviation {pair.get('angle_deviation_deg', 0):.1f}° from wind direction",
                        "rule": "turbine_pair_alignment"
                    })
                    if severity == "info":
                        severity = "warning"
        
        # If no issues found
        if not findings:
            findings.append({
                "type": "info",
                "message": "All rule-based checks passed. Weather conditions and yaw recommendations are within expected ranges.",
                "rule": "all_checks_passed"
            })
        
        # LLM-based expert review (if available)
        llm_assessment = None
        if self.llm:
            try:
                llm_assessment = await self._llm_review_agent2(
                    weather_data, expert_analysis, agent2b_result, turbine_pairs, findings
                )
            except Exception as e:
                logger.error(f"LLM review failed for Agent 2: {e}")
                findings.append({
                    "type": "info",
                    "message": f"LLM expert review unavailable (error: {str(e)}). Relying on rule-based validation only.",
                    "rule": "llm_fallback"
                })
        
        # Determine if workflow can continue
        allow_continue = self._should_allow_continue(severity)
        
        review = {
            "checkpoint": "checkpoint1_agent2",
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "findings": findings,
            "llm_assessment": llm_assessment,
            "allow_continue": allow_continue,
            "reviewer_mode": self.mode,
            "summary": self._generate_summary(findings, llm_assessment)
        }
        
        self.checkpoint_reviews.append(review)
        logger.info(f"Checkpoint 1 complete: severity={severity}, allow_continue={allow_continue}")
        
        return review
    
    async def review_agent3(
        self,
        weather_data: Dict[str, Any],
        expert_analysis: Dict[str, Any],
        power_prediction: Dict[str, Any],
        optimization_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Review Agent 3 outputs (power predictions and optimization).
        
        Validates:
        - Power predictions within physically plausible range
        - Uncertainty bounds reasonable
        - Optimization justified by wind conditions
        - Yaw angles within training range
        - Power gains realistic
        
        Args:
            weather_data: Weather conditions
            expert_analysis: Agent 2 outputs
            power_prediction: Agent 3 power predictions
            optimization_result: Optional optimization outputs
        
        Returns:
            Review dict with findings and severity
        """
        if not self.enabled:
            return self._disabled_review("checkpoint2_agent3")
        
        logger.info("Starting Checkpoint 2: Agent 3 Review")
        
        findings = []
        severity = "info"
        
        wind_speed = weather_data.get("wind_speed_ms", 0)
        
        # Check 1: Power range validation
        if "summary" in power_prediction:
            mean_power = power_prediction["summary"].get("mean_power_MW", 0)
            max_power = power_prediction["summary"].get("max_power_MW", 0)
            mean_uncertainty = power_prediction["summary"].get("mean_uncertainty_MW", 0)
            
            if mean_power < 0:
                findings.append({
                    "type": "critical",
                    "message": f"Predicted mean power is NEGATIVE ({mean_power:.2f} MW). This is physically impossible.",
                    "rule": "negative_power"
                })
                severity = "critical"
            
            if max_power > self.TURBINE_SPECS["rated_power_mw"] * 1.1:
                findings.append({
                    "type": "critical",
                    "message": f"Predicted max power {max_power:.2f} MW exceeds rated capacity ({self.TURBINE_SPECS['rated_power_mw']} MW) by >10%. Check model validity.",
                    "rule": "power_exceeds_rated"
                })
                severity = "critical"
            
            # Check uncertainty bounds
            if mean_uncertainty > mean_power * 0.5:
                findings.append({
                    "type": "warning",
                    "message": f"High uncertainty ({mean_uncertainty:.2f} MW) relative to mean power ({mean_power:.2f} MW). Predictions may be unreliable.",
                    "rule": "high_uncertainty"
                })
                if severity == "info":
                    severity = "warning"
        
        # Check 2: Optimization validation
        if optimization_result:
            # Check if optimization should have run
            if wind_speed < self.TURBINE_SPECS["cut_in_speed_ms"]:
                findings.append({
                    "type": "critical",
                    "message": f"Wake steering optimization ran with wind speed {wind_speed:.2f} m/s BELOW cut-in. Optimization is meaningless when turbine is not operating.",
                    "rule": "optimization_below_cutoff"
                })
                severity = "critical"
            
            # Check power gain plausibility
            if "power_gain_mw" in optimization_result:
                gain = optimization_result["power_gain_mw"]
                baseline_power = optimization_result.get("baseline_power_mw", 1.0)
                
                if baseline_power > 0:
                    gain_percent = (gain / baseline_power) * 100
                    if gain_percent > 20:
                        findings.append({
                            "type": "warning",
                            "message": f"Optimization reports {gain_percent:.1f}% power gain. Gains >20% are uncommon in wake steering and should be verified.",
                            "rule": "excessive_power_gain"
                        })
                        if severity == "info":
                            severity = "warning"
        
        # Check 3: Yaw angle extrapolation
        yaw_angle = expert_analysis.get("suggested_yaw", 0) or power_prediction.get("yaw_angle", 0)
        # Note: Training range check would require access to model metadata
        # For now, we flag very high yaw misalignments
        yaw_misalignment = expert_analysis.get("yaw_misalignment", 0)
        if abs(yaw_misalignment) > self.TURBINE_SPECS["max_yaw_misalignment_deg"]:
            findings.append({
                "type": "warning",
                "message": f"Yaw misalignment {abs(yaw_misalignment):.1f}° exceeds typical range (0-{self.TURBINE_SPECS['max_yaw_misalignment_deg']}°). Model may be extrapolating.",
                "rule": "yaw_extrapolation"
            })
            if severity == "info":
                severity = "warning"
        
        if not findings:
            findings.append({
                "type": "info",
                "message": "Power predictions are within expected ranges and physically plausible.",
                "rule": "all_checks_passed"
            })
        
        # LLM expert review
        llm_assessment = None
        if self.llm:
            try:
                llm_assessment = await self._llm_review_agent3(
                    weather_data, expert_analysis, power_prediction, optimization_result, findings
                )
            except Exception as e:
                logger.error(f"LLM review failed for Agent 3: {e}")
                findings.append({
                    "type": "info",
                    "message": f"LLM expert review unavailable. Rule-based validation only.",
                    "rule": "llm_fallback"
                })
        
        allow_continue = self._should_allow_continue(severity)
        
        review = {
            "checkpoint": "checkpoint2_agent3",
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "findings": findings,
            "llm_assessment": llm_assessment,
            "allow_continue": allow_continue,
            "reviewer_mode": self.mode,
            "summary": self._generate_summary(findings, llm_assessment)
        }
        
        self.checkpoint_reviews.append(review)
        logger.info(f"Checkpoint 2 complete: severity={severity}, allow_continue={allow_continue}")
        
        return review
    
    async def review_agent4(
        self,
        weather_data: Dict[str, Any],
        expert_analysis: Dict[str, Any],
        wake_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review Agent 4 outputs (wake flow predictions).
        
        Validates:
        - Velocity fields physically reasonable
        - Wake deficit consistent with conditions
        - Velocity statistics align with upstream wind
        
        Args:
            weather_data: Weather conditions
            expert_analysis: Agent 2 outputs
            wake_prediction: Agent 4 wake predictions
        
        Returns:
            Review dict with findings and severity
        """
        if not self.enabled:
            return self._disabled_review("checkpoint3_agent4")
        
        logger.info("Starting Checkpoint 3: Agent 4 Review")
        
        findings = []
        severity = "info"
        
        wind_speed = weather_data.get("wind_speed_ms", 0)
        
        # Check 1: Velocity magnitude validation
        if "velocity_magnitude" in wake_prediction:
            vel_mag = wake_prediction["velocity_magnitude"]
            
            # Check for impossible values (would need numpy analysis)
            # For now, perform basic sanity checks on available statistics
            findings.append({
                "type": "info",
                "message": f"Wake flow prediction generated for wind speed {wind_speed:.2f} m/s. Detailed velocity field validation requires array analysis.",
                "rule": "wake_generated"
            })
        
        # Check 2: Wake feasibility check
        yaw_misalignment = expert_analysis.get("yaw_misalignment", 0)
        if abs(yaw_misalignment) > 0:
            findings.append({
                "type": "info",
                "message": f"Wake steering active with {abs(yaw_misalignment):.1f}° yaw misalignment. Wake deflection expected in flow field.",
                "rule": "wake_steering_active"
            })
        
        if not findings:
            findings.append({
                "type": "info",
                "message": "Wake flow prediction completed. Detailed validation requires visualization.",
                "rule": "wake_completed"
            })
        
        # LLM expert review
        llm_assessment = None
        if self.llm:
            try:
                llm_assessment = await self._llm_review_agent4(
                    weather_data, expert_analysis, wake_prediction, findings
                )
            except Exception as e:
                logger.error(f"LLM review failed for Agent 4: {e}")
                findings.append({
                    "type": "info",
                    "message": f"LLM expert review unavailable. Basic validation only.",
                    "rule": "llm_fallback"
                })
        
        allow_continue = self._should_allow_continue(severity)
        
        review = {
            "checkpoint": "checkpoint3_agent4",
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "findings": findings,
            "llm_assessment": llm_assessment,
            "allow_continue": allow_continue,
            "reviewer_mode": self.mode,
            "summary": self._generate_summary(findings, llm_assessment)
        }
        
        self.checkpoint_reviews.append(review)
        logger.info(f"Checkpoint 3 complete: severity={severity}, allow_continue={allow_continue}")
        
        return review
    
    async def generate_final_review(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive final review synthesizing all checkpoint reviews.
        
        Args:
            results: Complete results dict from orchestrator
        
        Returns:
            Final review with overall assessment and recommendations
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "Reviewer not enabled"
            }
        
        logger.info("Generating final review synthesis")
        
        # Count findings by severity across all checkpoints
        critical_count = 0
        warning_count = 0
        info_count = 0
        
        all_findings = []
        
        for review in self.checkpoint_reviews:
            for finding in review.get("findings", []):
                finding_type = finding.get("type", "info")
                all_findings.append(finding)
                
                if finding_type == "critical":
                    critical_count += 1
                elif finding_type == "warning":
                    warning_count += 1
                else:
                    info_count += 1
        
        # Determine overall status
        if critical_count > 0:
            overall_status = "FAILED"
            status_message = f"Analysis contains {critical_count} critical issue(s) requiring attention."
        elif warning_count > 0:
            overall_status = "WARNING"
            status_message = f"Analysis completed with {warning_count} warning(s)."
        else:
            overall_status = "APPROVED"
            status_message = "Analysis passed all validation checks."
        
        # Extract key findings (critical and warnings only)
        key_findings = [
            f for f in all_findings 
            if f.get("type") in ["critical", "warning"]
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(key_findings, results)
        
        final_review = {
            "overall_status": overall_status,
            "status_message": status_message,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "key_findings": key_findings,
            "recommendations": recommendations,
            "checkpoint_count": len(self.checkpoint_reviews),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Final review: {overall_status}, {critical_count} critical, {warning_count} warnings")
        
        return final_review
    
    # ==================== Private Helper Methods ====================
    
    def _should_allow_continue(self, severity: str) -> bool:
        """Determine if workflow should continue based on severity and mode."""
        if self.mode == "advisory":
            return True  # Always allow in advisory mode
        elif self.mode == "blocking":
            return severity != "critical"  # Block only on critical
        return True
    
    def _disabled_review(self, checkpoint: str) -> Dict[str, Any]:
        """Return dummy review when reviewer is disabled."""
        return {
            "checkpoint": checkpoint,
            "status": "disabled",
            "severity": "info",
            "findings": [],
            "allow_continue": True,
            "message": "Reviewer not enabled"
        }
    
    def _generate_summary(self, findings: List[Dict], llm_assessment: Optional[str]) -> str:
        """Generate human-readable summary of review."""
        critical = [f for f in findings if f.get("type") == "critical"]
        warnings = [f for f in findings if f.get("type") == "warning"]
        
        if critical:
            summary = f"Found {len(critical)} critical issue(s). "
        elif warnings:
            summary = f"Found {len(warnings)} warning(s). "
        else:
            summary = "All checks passed. "
        
        if llm_assessment:
            summary += "LLM expert provided additional assessment."
        
        return summary
    
    def _generate_recommendations(
        self, 
        key_findings: List[Dict], 
        results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        # Check for common issues
        finding_rules = [f.get("rule") for f in key_findings]
        
        if "cut_in_violation" in finding_rules or "cut_out_violation" in finding_rules:
            recommendations.append(
                "⚠️ Wind speed outside operational range. Consider waiting for better conditions."
            )
        
        if "optimization_below_cutoff" in finding_rules:
            recommendations.append(
                "🔴 CRITICAL: Do not proceed with optimization when turbine is not operating. Wait for wind speed to exceed cut-in."
            )
        
        if "power_exceeds_rated" in finding_rules or "negative_power" in finding_rules:
            recommendations.append(
                "🔴 CRITICAL: Power predictions are physically impossible. Review model calibration and input data."
            )
        
        if "high_uncertainty" in finding_rules:
            recommendations.append(
                "⚠️ High prediction uncertainty detected. Consider collecting more data or adjusting model parameters."
            )
        
        if "yaw_extrapolation" in finding_rules:
            recommendations.append(
                "⚠️ Operating outside model training range. Predictions may be unreliable. Validate with field data."
            )
        
        if "turbine_pair_alignment" in finding_rules:
            recommendations.append(
                "⚠️ Some turbine pairs show misalignment with wind direction. Review pair selection or wait for wind shift."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ Analysis looks good. No critical issues detected. Proceed with confidence."
            )
        
        return recommendations
    
    async def _llm_review_agent2(
        self,
        weather_data: Dict[str, Any],
        expert_analysis: Dict[str, Any],
        agent2b_result: Optional[Dict[str, Any]],
        turbine_pairs: Optional[Dict[str, Any]],
        rule_findings: List[Dict]
    ) -> Optional[str]:
        """LLM-based expert review for Agent 2."""
        
        system_prompt = """You are a wind turbine expert with deep knowledge of:
- NREL 5MW turbine specifications and operational constraints
- Wind turbine control strategies (yaw, pitch)
- Wake effects and turbine interactions
- Power generation optimization
- Safety protocols

Your role is to review wind farm analysis outputs and identify issues."""

        # Build context
        context = f"""
TURBINE: NREL 5MW Reference Turbine
- Rated Power: 5.0 MW
- Cut-in Wind Speed: 3.0 m/s
- Cut-out Wind Speed: 25.0 m/s
- Rated Wind Speed: 11.4 m/s
- Rotor Diameter: 126 m
- Hub Height: 90 m

CURRENT WEATHER CONDITIONS:
- Wind Speed: {weather_data.get('wind_speed_ms', 0):.2f} m/s
- Wind Direction: {weather_data.get('wind_direction_deg', 0):.1f}°
- Temperature: {weather_data.get('temperature_c', 'N/A')}°C

AGENT 2 EXPERT RECOMMENDATIONS:
- Suggested Yaw Angle: {expert_analysis.get('suggested_yaw', 0):.1f}°
- Yaw Misalignment: {expert_analysis.get('yaw_misalignment', 0):.1f}°
- Operating Region: {expert_analysis.get('operating_region', 'Unknown')}
- Expected Efficiency: {expert_analysis.get('expected_efficiency', 0):.1%}
- Reasoning: {', '.join(expert_analysis.get('reasoning', []))}
"""

        if agent2b_result:
            context += f"\n\nLLM EXPERT (Agent 2B) OPINION:\n{agent2b_result.get('recommendation', 'N/A')}\n"
        
        if turbine_pairs and turbine_pairs.get("status") == "success":
            pairs = turbine_pairs.get("turbine_pairs", [])
            context += f"\n\nTURBINE PAIR ANALYSIS ({len(pairs)} pairs identified):\n"
            for pair in pairs[:3]:  # Show first 3 pairs
                context += f"- Turbine {pair.get('upstream_turbine')} -> {pair.get('downstream_turbine')}: "
                context += f"Distance={pair.get('distance_km', 0):.2f} km, "
                context += f"Bearing={pair.get('bearing_deg', 0):.1f}°, "
                context += f"Wake Strength={pair.get('wake_strength', 'unknown')}\n"
        
        context += f"\n\nRULE-BASED VALIDATION RESULTS:\n"
        for finding in rule_findings:
            context += f"- [{finding['type'].upper()}] {finding['message']}\n"
        
        prompt = f"""{context}

TASK: Review the Agent 2 outputs as a wind turbine expert. Consider:
1. Are weather conditions suitable for turbine operation?
2. Is the yaw angle recommendation appropriate for these conditions?
3. If turbine pairs were identified, do they make sense given the wind direction?
4. Do you agree with the operating region classification?
5. Are there any safety or operational concerns?

Provide a concise expert assessment (max 150 words) focusing on validation and any concerns.
"""
        
        try:
            assessment = await self.llm.complete(
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            logger.info("LLM review for Agent 2 completed")
            return assessment.strip()
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            return None
    
    async def _llm_review_agent3(
        self,
        weather_data: Dict[str, Any],
        expert_analysis: Dict[str, Any],
        power_prediction: Dict[str, Any],
        optimization_result: Optional[Dict[str, Any]],
        rule_findings: List[Dict]
    ) -> Optional[str]:
        """LLM-based expert review for Agent 3."""
        
        system_prompt = """You are a wind turbine expert specializing in power performance and optimization.
Review power predictions and optimization results for physical plausibility and consistency."""

        summary = power_prediction.get("summary", {})
        
        context = f"""
WEATHER & CONTROL:
- Wind Speed: {weather_data.get('wind_speed_ms', 0):.2f} m/s
- Yaw Angle: {expert_analysis.get('suggested_yaw', power_prediction.get('yaw_angle', 0)):.1f}°
- Yaw Misalignment: {expert_analysis.get('yaw_misalignment', 0):.1f}°

POWER PREDICTION RESULTS:
- Mean Power: {summary.get('mean_power_MW', 0):.3f} MW
- Max Power: {summary.get('max_power_MW', 0):.3f} MW
- Min Power: {summary.get('min_power_MW', 0):.3f} MW
- Mean Uncertainty: {summary.get('mean_uncertainty_MW', 0):.3f} MW
"""

        if optimization_result:
            context += f"""
OPTIMIZATION RESULTS:
- Method: {optimization_result.get('method', 'N/A')}
- Baseline Power: {optimization_result.get('baseline_power_mw', 0):.3f} MW
- Optimized Power: {optimization_result.get('optimized_power_mw', 0):.3f} MW
- Power Gain: {optimization_result.get('power_gain_mw', 0):.3f} MW ({optimization_result.get('power_gain_percent', 0):.1f}%)
"""
        
        context += f"\n\nRULE-BASED VALIDATION:\n"
        for finding in rule_findings:
            context += f"- [{finding['type'].upper()}] {finding['message']}\n"
        
        prompt = f"""{context}

TASK: Review the power predictions and optimization as an expert. Consider:
1. Are power values physically plausible given wind speed and yaw angle?
2. Is the uncertainty level reasonable?
3. If optimization ran, is the power gain realistic?
4. Are there any red flags or concerns?

Provide a concise assessment (max 150 words).
"""
        
        try:
            assessment = await self.llm.complete(
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            logger.info("LLM review for Agent 3 completed")
            return assessment.strip()
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            return None
    
    async def _llm_review_agent4(
        self,
        weather_data: Dict[str, Any],
        expert_analysis: Dict[str, Any],
        wake_prediction: Dict[str, Any],
        rule_findings: List[Dict]
    ) -> Optional[str]:
        """LLM-based expert review for Agent 4."""
        
        system_prompt = """You are a wind turbine expert specializing in wake aerodynamics.
Review wake flow predictions for physical consistency."""

        context = f"""
CONDITIONS:
- Wind Speed: {weather_data.get('wind_speed_ms', 0):.2f} m/s
- Yaw Angle: {expert_analysis.get('suggested_yaw', 0):.1f}°
- Yaw Misalignment: {expert_analysis.get('yaw_misalignment', 0):.1f}°

WAKE PREDICTION:
- Timesteps: {wake_prediction.get('timesteps', 'N/A')}
- Spatial Points: {wake_prediction.get('spatial_points', 'N/A')}
- Status: {wake_prediction.get('status', 'unknown')}
"""

        context += f"\n\nVALIDATION NOTES:\n"
        for finding in rule_findings:
            context += f"- {finding['message']}\n"
        
        prompt = f"""{context}

TASK: Brief expert assessment of wake flow prediction (max 100 words).
Consider physical consistency and wake steering expectations.
"""
        
        try:
            assessment = await self.llm.complete(
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            logger.info("LLM review for Agent 4 completed")
            return assessment.strip()
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            return None
